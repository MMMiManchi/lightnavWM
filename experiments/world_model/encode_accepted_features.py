"""Encode RGB observations from an approved fit/dev run set into a frozen cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.world_model.cache_writer import (
    FEATURE_CACHE_ROOT,
    _refuse_destination,
    verify_bound_files,
)
from experiments.world_model.feature_cache import (
    AUDITED_PREPROCESSOR,
    FEATURE_CACHE_VERSION,
    IDENTITY_FIELDS,
    OUTPUT_SHAPES,
    SIGLIP_ENCODER_ID,
    SIGLIP_REVISION,
    FeatureContractError,
    check_feature_cache,
    feature_checksum,
)
from experiments.world_model.jsonutil import dumps, sha256_file
from experiments.world_model.latent_data import (
    load_accepted_run_set,
    read_admitted_rgb,
    scene_partition,
)
from experiments.world_model.training import materialize_feature_samples


def _read_approval_pairs(values: list[str]) -> list[tuple[Path, Path]]:
    pairs = []
    for value in values:
        run_text, separator, approval_text = value.partition("=")
        if not separator or not run_text or not approval_text:
            raise ValueError("each --approved-run must be RUN_DIR=APPROVAL_JSON")
        pairs.append((Path(run_text), Path(approval_text)))
    return pairs


def _cache_manifest(weight_path: Path) -> dict[str, Any]:
    bound = verify_bound_files(weight_path)
    return {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "feature_layer_selected": None,
        **IDENTITY_FIELDS,
        "encoder_id": SIGLIP_ENCODER_ID,
        "revision": SIGLIP_REVISION,
        "preprocess_path": str(AUDITED_PREPROCESSOR.resolve()),
        "weight_path": str(weight_path.resolve()),
        "weight_sha256": bound["weight_sha256"],
        "preprocess_sha256": bound["preprocess_sha256"],
        "output_name": "pooler_output",
        "output_shape": list(OUTPUT_SHAPES["pooler_output"]),
    }


def _encode_rows(bundle: dict[str, Any], *, snapshot: Path, batch_size: int, device: str) -> list[dict[str, Any]]:
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel

    if not torch.cuda.is_available() or device != "cuda:0":
        raise RuntimeError("formal feature encoding requires the explicitly selected CUDA device cuda:0")
    if torch.version.cuda != "12.8" or torch.__version__ != "2.10.0+cu128":
        raise RuntimeError(f"unapproved torch/CUDA runtime: {torch.__version__} / {torch.version.cuda}")
    if __import__("transformers").__version__ != "5.8.0":
        raise RuntimeError("transformers version does not match the audited feature contract")

    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    model = SiglipVisionModel.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device).eval()
    episode_by_uid = {str(row["episode_uid"]): row for row in bundle["episodes"]}
    observations = sorted(bundle["observations"], key=lambda row: (str(row["episode_uid"]), int(row["step_index"])))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(observations), batch_size):
        chunk = observations[start : start + batch_size]
        images = []
        metadata = []
        for observation in chunk:
            frame = observation.get("rgb")
            if frame is None or observation.get("rgb_missing"):
                raise FeatureContractError("missing_image", str(observation.get("observation_id")))
            relative = str(frame["relative_path"])
            run_dir = next(
                (path for path in bundle["run_dirs"] if f"{path}:{relative}" in bundle["admitted_frames"]),
                None,
            )
            if run_dir is None:
                raise FeatureContractError("missing_image", relative)
            episode = episode_by_uid[str(observation["episode_uid"])]
            images.append(read_admitted_rgb(frame))
            metadata.append(
                {
                    "episode_uid": str(observation["episode_uid"]),
                    "step_index": int(observation["step_index"]),
                    "observation_id": str(observation["observation_id"]),
                    "image_path": relative,
                    "image_sha256": str(frame["sha256"]),
                    "source_split": "train",
                    "source_run": str(run_dir),
                    "scene": str(episode["scene"]),
                    "route_group_id": str(episode["route_group_id"]),
                    "scene_partition": scene_partition(str(episode["scene"])),
                    "revision": SIGLIP_REVISION,
                    "weight_sha256": IDENTITY_FIELDS["weight_sha256"],
                    "preprocess_sha256": IDENTITY_FIELDS["preprocess_sha256"],
                    "output_name": "pooler_output",
                    "feature_dtype": "float32",
                    "feature_shape": list(OUTPUT_SHAPES["pooler_output"]),
                }
            )
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            output = model(pixel_values=pixels).pooler_output.float().cpu().numpy()
        if output.shape != (len(chunk), *OUTPUT_SHAPES["pooler_output"]) or not np.isfinite(output).all():
            raise FeatureContractError("feature_dim_mismatch", f"unexpected encoder output {output.shape}")
        for base, vector in zip(metadata, output, strict=True):
            values = np.asarray(vector, dtype=np.float32).reshape(-1).tolist()
            rows.append({**base, "feature_sha256": feature_checksum(values), "feature": values})
    return rows


def _write_cache(
    destination: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    _refuse_destination(destination)
    destination.mkdir(parents=True, exist_ok=False)
    assets = destination / "assets"
    assets.mkdir()
    linked: dict[str, str] = {}
    for row in rows:
        relative = str(row["image_path"])
        source = Path(str(row["source_run"])) / relative
        previous = linked.get(relative)
        if previous is not None:
            if previous != str(row["image_sha256"]):
                raise FeatureContractError("image_hash_mismatch", f"duplicate cache path {relative}")
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != row["image_sha256"]:
            raise FeatureContractError("image_hash_mismatch", f"copied image differs: {relative}")
        linked[relative] = str(row["image_sha256"])
    files = {
        "manifest.json": dumps(manifest) + "\n",
        "features.jsonl": "".join(dumps(row) + "\n" for row in rows),
    }
    try:
        for name, text in files.items():
            fd = os.open(destination / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception:
        # Preserve any partial output for diagnosis; the directory is never reused.
        raise
    for path in destination.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)


def _make_validation_image_root(bundle: dict[str, Any], root: Path) -> Path:
    assets = root / "assets"
    assets.mkdir(parents=True)
    seen: dict[str, str] = {}
    for run_dir in bundle["run_dirs"]:
        prefix = f"{run_dir}:"
        for key, frame in bundle["admitted_frames"].items():
            if not key.startswith(prefix):
                continue
            relative = str(frame["relative_path"])
            digest = str(frame["sha256"])
            previous = seen.get(relative)
            if previous is not None:
                if previous != digest:
                    raise FeatureContractError("image_hash_mismatch", f"duplicate cache path {relative}")
                continue
            seen[relative] = digest
            source = (run_dir / relative).resolve()
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encode approved train_fit/train_dev RGB observations")
    parser.add_argument("--approved-run", action="append", required=True, metavar="RUN_DIR=APPROVAL_JSON")
    parser.add_argument("--approval-key", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.output_name != "pooler_output":
        raise SystemExit("the v1 predictor currently supports only pooler_output")
    if args.device != "cuda:0":
        raise SystemExit("the audited feature extraction path is pinned to cuda:0")

    pairs = _read_approval_pairs(args.approved_run)
    config_identities = set()
    for run_dir, _approval in pairs:
        run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        config_record = run_manifest.get("config") or {}
        config_path = (Path.cwd() / str(config_record.get("path", ""))).resolve()
        try:
            config_path.relative_to(Path.cwd().resolve())
        except ValueError as exc:
            raise SystemExit(f"run config escapes repository: {config_path}") from exc
        if not config_path.is_file() or sha256_file(config_path) != config_record.get("sha256"):
            raise SystemExit(f"run config no longer matches its admitted snapshot: {config_path}")
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        agent = config["habitat"]["simulator"]["agents"]["main_agent"]
        if float(agent["radius"]) != 0.10 or float(agent["height"]) != 0.50:
            raise SystemExit(f"run geometry is not the selected Go2-W 0.10 m configuration: {config_path}")
        config_identities.add(str(config_record["sha256"]))
    if len(config_identities) != 1:
        raise SystemExit("approved fit/dev runs do not share one identical 0.10 m config snapshot")
    sources = [(run, approval, args.approval_key) for run, approval in pairs]
    bundle = load_accepted_run_set(sources)
    weight_path = args.snapshot / "model.safetensors"
    manifest = _cache_manifest(weight_path)
    features = _encode_rows(bundle, snapshot=args.snapshot, batch_size=args.batch_size, device=args.device)
    split_episodes = bundle["formal_split"]["episodes"]
    with tempfile.TemporaryDirectory(prefix="wm-feature-verify-") as temporary_root:
        image_root = _make_validation_image_root(bundle, Path(temporary_root))
        check_feature_cache(features, manifest, image_root=image_root, split_episodes=split_episodes)
        fit, dev = materialize_feature_samples(
            bundle,
            features,
            manifest,
            image_root=image_root,
            split_episodes=split_episodes,
            output_name=args.output_name,
        )
    destination = FEATURE_CACHE_ROOT / args.output_name
    _write_cache(destination, manifest, features)
    print(
        json.dumps(
            {
                "cache_dir": str(destination),
                "feature_rows": len(features),
                "paired_fit_transitions": len(fit),
                "paired_dev_transitions": len(dev),
                "formal_training_started": False,
                "formal_evaluation_started": False,
                "feature_layer_selected": None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
