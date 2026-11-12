#!/usr/bin/env python3
"""LoRA-tune LightNav-0 on the Go2-W expert and correction samples.

The prompt, frame resize and post-ViT pooling match inference. Loss is the
usual next-token loss, which the data processor restricts to the assistant
action tokens. Vision stays frozen; LoRA is on the language-model attention
projections, which are the modules that choose the action.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from lightnav.expert import trim_episode_rows  # noqa: E402
from lightnav.inference.config import InferenceConfig  # noqa: E402
from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor  # noqa: E402
from lightnav.inference.model import load_hf_model  # noqa: E402
from lightnav.inference.samples import _build_slowfast_eval_sample  # noqa: E402


def _rows(dataset: Path) -> tuple[list[dict], int]:
    raw = []
    paths = [dataset / "samples.jsonl"]
    paths += sorted(dataset.glob("corrections*.jsonl"))
    for path in paths:
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("action") and row.get("frame_dir") is not None and "step" in row:
                    raw.append(row)
    episodes: dict[tuple, list[dict]] = {}
    episode_order: list[tuple] = []
    for row in raw:
        key = (row.get("source"), row.get("scene"), str(row.get("episode_id")))
        if key not in episodes:
            episode_order.append(key)
            episodes[key] = []
        episodes[key].append(row)
    rows: list[dict] = []
    dropped = 0
    for key in episode_order:
        trimmed = trim_episode_rows(episodes[key])
        dropped += len(episodes[key]) - len(trimmed)
        rows.extend(trimmed)
    if not rows:
        raise FileNotFoundError(f"no trainable rows under {dataset}")
    return rows, dropped


def _videos(dataset: Path, rows: list[dict], video_size: tuple[int, int]):
    """Yield ``(row, video)`` loading each episode's frames once, in step order."""
    cache_key = None
    frames: list[torch.Tensor] = []
    for row in rows:
        key = (row.get("source"), row.get("scene"), str(row.get("episode_id")), row["frame_dir"])
        if key != cache_key:
            frames = []
            cache_key = key
        frame_dir = dataset / row["frame_dir"]
        need = int(row["step"]) + 1
        while len(frames) < need:
            path = frame_dir / f"{len(frames):04d}.jpg"
            if not path.is_file():
                raise FileNotFoundError(path)
            frames.append(rgb_frame_to_model_tensor(iio.imread(path), video_size))
        yield row, torch.stack(frames[:need], dim=0)


def _batch(bundle, row: dict, video: torch.Tensor, device: torch.device) -> dict:
    sample = _build_slowfast_eval_sample(video, row["instruction"], None, bundle)
    sample["conversations"][1]["value"] = row["action"]
    processed = bundle.data_processor.process_sample(sample)
    original_grid = processed.pop("_original_video_grid_thw", None)
    pool_factors = processed.pop("_post_vit_pool_factors", None)
    processed.pop("_post_vit_pool_spatial", None)
    processed.pop("position_ids", None)
    processed.pop("image_mask", None)
    processed.pop("video_mask", None)
    target = getattr(bundle.model, "_post_vit_target", None)
    if target is not None and original_grid is not None:
        target._post_vit_original_grid = original_grid.to(device)
        if torch.is_tensor(pool_factors):
            target._post_vit_pool_factors = pool_factors.to(device)
        else:
            target._post_vit_pool_factors = pool_factors
    batch = {}
    for key in (
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    ):
        value = processed.get(key)
        if value is None:
            continue
        if key in ("input_ids", "attention_mask", "labels") and value.ndim == 1:
            value = value.unsqueeze(0)
        if value.is_floating_point():
            value = value.to(device=device, dtype=torch.bfloat16)
        else:
            value = value.to(device)
        batch[key] = value
    # Stock Qwen3-VL recomputes mrope and needs these masks once position_ids
    # have been dropped. 1 = image pad, 2 = video pad.
    ids = batch["input_ids"]
    mm = torch.zeros_like(ids)
    mm[ids == bundle.data_processor.image_token_id] = 1
    mm[ids == bundle.data_processor.video_token_id] = 2
    batch["mm_token_type_ids"] = mm
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="output/go2w_expert")
    parser.add_argument("--model", default="checkpoints/LightNav-0")
    parser.add_argument("--out", default="checkpoints/LightNav-0-go2w")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 trains on every row.")
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    rows, dropped = _rows(Path(args.dataset))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    print(f"[train] samples={len(rows)} stalled_steps_dropped={dropped}", flush=True)

    bundle = load_hf_model(
        InferenceConfig(model_path=args.model, backend="hf", device="cuda"),
        task_type="vlnce",
    )
    base = bundle.model
    post_vit_target = getattr(base, "_post_vit_target", None)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    out = Path(args.out)
    state_path = out / "trainer_state.json"
    resume = state_path.is_file() and (out / "adapter_config.json").is_file()
    if resume:
        model = PeftModel.from_pretrained(base, out, is_trainable=True)
        state = json.loads(state_path.read_text())
        print(f"[train] resume epoch {state['epoch']} sample {state['next_index']}", flush=True)
    else:
        model = get_peft_model(base, lora)
        state = {"epoch": 0, "next_index": 0}
    model._post_vit_target = post_vit_target
    model.config.use_cache = False
    bundle.model = model
    model.print_trainable_parameters()
    model.train()
    optim = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    optim_path = out / "optimizer.pt"
    if resume and optim_path.is_file():
        optim.load_state_dict(torch.load(optim_path, map_location=bundle.device))

    device = bundle.device
    video_size = tuple(bundle.video_size)
    step = 0
    optim.zero_grad(set_to_none=True)

    def save(next_index: int, epoch: int) -> None:
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        bundle.processor.save_pretrained(out)
        torch.save(optim.state_dict(), optim_path)
        state_path.write_text(json.dumps({"epoch": epoch, "next_index": next_index}) + "\n")

    for epoch in range(int(state["epoch"]), args.epochs):
        total = 0.0
        start = int(state["next_index"]) if epoch == int(state["epoch"]) else 0
        epoch_rows = rows[start:]
        seen = 0
        for offset, (row, video) in enumerate(_videos(Path(args.dataset), epoch_rows, video_size)):
            index = start + offset
            batch = _batch(bundle, row, video, device)
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            total += float(loss.detach()) * args.grad_accum
            seen += 1
            if seen % args.grad_accum == 0 or index + 1 == len(rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                print(
                    f"[train] epoch {epoch} sample {index + 1}/{len(rows)} "
                    f"loss {total / seen:.4f} step {step} "
                    f"peak_gib {torch.cuda.max_memory_allocated() / 2**30:.2f}",
                    flush=True,
                )
                if step % 100 == 0:
                    save(index + 1, epoch)
                    print(f"[train] checkpoint sample {index + 1}", flush=True)
        save(0, epoch + 1)
    print(f"[train] saved {out}", flush=True)


if __name__ == "__main__":
    main()
