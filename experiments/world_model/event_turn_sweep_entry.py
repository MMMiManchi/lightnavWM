"""Score a turn by the frames recorded while turning, not by the settled frame alone.

Writes a diagnostic score. Does not train, collect, or write the formal feature cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.world_model.cache_writer import verify_bound_files
from experiments.world_model.ego_warp_entry import index_observations, load_checked
from experiments.world_model.event_place import (
    LONG_PATH_M,
    SWEEP_MARGIN,
    cosine,
    mean_normalized,
    other_span_indexes,
    sweep_verdict,
)
from experiments.world_model.event_spans import start_view_ids
from experiments.world_model.event_spans_entry import SPLITS, _spans_for_split
from experiments.world_model.feature_cache import (
    AUDITED_PREPROCESSOR,
    OUTPUT_SHAPES,
    SIGLIP_REVISION,
    SIGLIP_WEIGHT_SHA256,
)

REPO = Path(__file__).resolve().parents[2]
SAVED_SPANS = REPO / "output/world_model_diagnostics/event_spans_20261103_01/event_spans.jsonl"
SAVED_SCORES = REPO / "output/world_model_diagnostics/event_place_20261103_01/place_scores.jsonl"
OUT = REPO / "output/world_model_diagnostics/event_turn_sweep_20261103_01"
RUNS = {
    "formal_dev_20261028_01": REPO / "output/world_model_data/formal_dev_20261028_01",
    "formal_fit_20261028_04": REPO / "output/world_model_data/formal_fit_20261028_04",
}
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots"
    / SIGLIP_REVISION
)
BATCH = 8
SINGLE_FRAME_TOLERANCE = 1e-3
_SPAN_KEYS = (
    "split",
    "episode_uid",
    "start_observation_id",
    "end_observation_id",
    "end_kind",
    "path_m",
    "recorded_control_steps",
)


def main() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {str(index) for index in range(2, 8)}:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to a single GPU from 2 to 7")
    spans = _load_spans()
    _require_saved_spans(spans)
    OUT.mkdir(parents=True, exist_ok=False)
    _write_json(OUT / "protocol.json", _protocol())
    observations = {split: index_observations(run) for split, run in RUNS.items()}
    needed = _needed_views(spans, observations)
    vectors = _encode(needed)
    others = other_span_indexes(spans)
    rows = [_score(spans, index, others[index], vectors) for index in range(len(spans))]
    drift = _single_frame_drift(rows)
    np.savez_compressed(
        OUT / "turn_view_pooler.npz",
        ids=np.array(["|".join(key) for key in vectors], dtype=object),
        features=np.stack([vectors[key] for key in vectors]),
    )
    _write_jsonl(OUT / "sweep_scores.jsonl", rows)
    group = [row for row in rows if row["end_kind"] == "turn" and row["path_m"] >= LONG_PATH_M]
    verdict = sweep_verdict(
        [row["cosine_start_end"] for row in group],
        [row["cosine_start_other"] for row in group],
    )
    summary = {
        "encoded_views": len(vectors),
        "span_count": len(rows),
        "verdict_count": len(group),
        "single_frame_max_abs_drift": drift,
        "groups": {
            "turn_at_least_1m": _group_summary(group),
        },
        "single_frame_turn_at_least_1m": _single_summary(group),
        "verdict": verdict,
    }
    _write_json(OUT / "results.json", summary)
    (OUT / "report.md").write_text(_report(summary), encoding="utf-8")


def _protocol() -> dict:
    return {
        "question": (
            "Does the mean of L2-normalized poolers over a turn bout separate the next "
            "node from another place by at least 0.10?"
        ),
        "node_descriptor": "mean of L2-normalized SigLIP vision-tower pooler vectors",
        "turn_bout": (
            "frames from the first step whose yaw exceeds the straight-step threshold "
            "through the settled frame; a slow turn keeps the short accumulation since "
            "the last straight-walk reset"
        ),
        "corridor_frames_included": False,
        "episode_start_and_episode_end": "one frame, because those nodes are not turns",
        "encoder": "SiglipVisionModel pooler_output",
        "output_shape": list(OUTPUT_SHAPES["pooler_output"]),
        "similarity": "cosine",
        "margin": SWEEP_MARGIN,
        "margin_fixed_after_single_frame_median_delta": 0.020,
        "margin_refit_to_this_run": False,
        "long_path_m": LONG_PATH_M,
        "verdict_group": "turn spans whose path is at least one meter",
        "other_place": "next span arrival in the same split, skipping this span's own views",
        "feature_layer_selected": None,
        "formal_cache_written": False,
        "training_started": False,
        "collection_started": False,
        "habitat_started": False,
        "branch_choice_tested": False,
    }


def _load_spans() -> list[dict]:
    spans = []
    for split in SPLITS:
        spans.extend(_spans_for_split(split, SPLITS[split]))
    starts = start_view_ids(spans)
    for span, start_ids in zip(spans, starts, strict=True):
        if span["end_kind"] == "turn" and len(span["end_view_ids"]) < 2:
            raise RuntimeError("a turn node needs the frames recorded during the turn")
        span["start_view_ids"] = start_ids
    return spans


def _require_saved_spans(spans: list[dict]) -> None:
    saved = [json.loads(line) for line in SAVED_SPANS.read_text(encoding="utf-8").splitlines() if line]
    if len(saved) != len(spans):
        raise RuntimeError(f"span count changed: {len(spans)} vs saved {len(saved)}")
    for fresh, previous in zip(spans, saved, strict=True):
        for key in _SPAN_KEYS:
            if fresh[key] != previous[key]:
                raise RuntimeError(f"span field {key} drifted: {fresh[key]} vs {previous[key]}")


def _needed_views(spans: list[dict], observations: dict[str, dict]) -> dict[tuple[str, str], dict]:
    needed: dict[tuple[str, str], dict] = {}
    for span in spans:
        for obs_id in [*span["start_view_ids"], *span["end_view_ids"]]:
            needed[(span["split"], obs_id)] = observations[span["split"]][obs_id]
    return needed


def _encode(needed: dict[tuple[str, str], dict]) -> dict[tuple[str, str], np.ndarray]:
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel

    weight = SNAPSHOT / "model.safetensors"
    if verify_bound_files(weight)["weight_sha256"] != SIGLIP_WEIGHT_SHA256:
        raise RuntimeError("SigLIP weight hash does not match the audited contract")
    if torch.version.cuda != "12.8" or torch.__version__ != "2.10.0+cu128":
        raise RuntimeError(f"unapproved torch/CUDA runtime: {torch.__version__}")
    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    model = (
        SiglipVisionModel.from_pretrained(SNAPSHOT, local_files_only=True, torch_dtype=torch.float32)
        .to("cuda:0")
        .eval()
    )
    keys = list(needed)
    features = np.empty((len(keys), *OUTPUT_SHAPES["pooler_output"]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(keys), BATCH):
            chunk_keys = keys[start : start + BATCH]
            images = [load_checked(RUNS[key[0]], needed[key]["rgb"]) for key in chunk_keys]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to("cuda:0")
            output = model(pixel_values=pixels).pooler_output.float().cpu().numpy()
            if output.shape != (len(chunk_keys), *OUTPUT_SHAPES["pooler_output"]) or not np.isfinite(output).all():
                raise RuntimeError(f"unexpected pooler output {output.shape}")
            features[start : start + len(chunk_keys)] = output
    return {key: features[index] for index, key in enumerate(keys)}


def _score(spans: list[dict], index: int, other_index: int, vectors: dict[tuple[str, str], np.ndarray]) -> dict:
    span = spans[index]
    other = spans[other_index]
    if other["split"] != span["split"]:
        raise RuntimeError("the other arrival crossed splits")
    start = mean_normalized(_stack(vectors, span["split"], span["start_view_ids"]))
    end = mean_normalized(_stack(vectors, span["split"], span["end_view_ids"]))
    other_vec = mean_normalized(_stack(vectors, other["split"], other["end_view_ids"]))
    single_start = vectors[(span["split"], span["start_observation_id"])]
    single_end = vectors[(span["split"], span["end_observation_id"])]
    single_other = vectors[(other["split"], other["end_observation_id"])]
    return {
        **span,
        "other_observation_id": other["end_observation_id"],
        "other_view_count": len(other["end_view_ids"]),
        "cosine_start_end": cosine(start, end),
        "cosine_start_other": cosine(start, other_vec),
        "cosine_single_start_end": cosine(single_start, single_end),
        "cosine_single_start_other": cosine(single_start, single_other),
    }


def _stack(vectors: dict[tuple[str, str], np.ndarray], split: str, obs_ids: list[str]) -> np.ndarray:
    return np.stack([vectors[(split, obs_id)] for obs_id in obs_ids])


def _single_frame_drift(rows: list[dict]) -> float:
    saved = [json.loads(line) for line in SAVED_SCORES.read_text(encoding="utf-8").splitlines() if line]
    previous = {
        (row["split"], row["episode_uid"], row["start_observation_id"], row["end_observation_id"]): row
        for row in saved
    }
    if len(previous) != len(saved) or len(previous) != len(rows):
        raise RuntimeError("saved single-frame scores do not match these spans")
    worst = 0.0
    for row in rows:
        key = (row["split"], row["episode_uid"], row["start_observation_id"], row["end_observation_id"])
        old = previous[key]
        worst = max(
            worst,
            abs(row["cosine_single_start_end"] - old["cosine_start_end"]),
            abs(row["cosine_single_start_other"] - old["cosine_start_other"]),
        )
    if worst > SINGLE_FRAME_TOLERANCE:
        raise RuntimeError(f"single-frame cosine drifted by {worst}")
    return worst


def _group_summary(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "median_cosine_start_end": _median([row["cosine_start_end"] for row in rows]),
        "median_cosine_start_other": _median([row["cosine_start_other"] for row in rows]),
        "median_delta": _median([row["cosine_start_end"] - row["cosine_start_other"] for row in rows]),
        "median_end_views": _median([float(len(row["end_view_ids"])) for row in rows]),
        "median_start_views": _median([float(len(row["start_view_ids"])) for row in rows]),
    }


def _single_summary(rows: list[dict]) -> dict[str, float | int]:
    return {
        "count": len(rows),
        "median_cosine_start_end": _median([row["cosine_single_start_end"] for row in rows]),
        "median_cosine_start_other": _median([row["cosine_single_start_other"] for row in rows]),
        "median_delta": _median(
            [row["cosine_single_start_end"] - row["cosine_single_start_other"] for row in rows]
        ),
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _report(summary: dict) -> str:
    verdict = summary["verdict"]
    group = summary["groups"]["turn_at_least_1m"]
    single = summary["single_frame_turn_at_least_1m"]
    if verdict["accepted"]:
        closing = "差距过了 0.10。这个合起来的描述可以当作地点钥匙。预测还没有开始。"
    else:
        closing = (
            "差距没有到 0.10。合起来的描述还是分不开下一地点和另一个地点，不能拿去训练。"
            "下一步才换和文字对齐的向量，并用同一条 0.10 再查。"
        )
    return "\n".join(
        [
            "# 转弯时的几帧能不能分开下一地点",
            "",
            "一个转弯节点用这次转弯过程里已经录下的画面合在一起，不用停下后的单独一张。",
            "走到转弯之前的走廊画面不放进节点。回合起点和回合结束不是转弯，仍然只用一张。",
            "",
            "门槛在算出结果之前定死为 0.10。它是在单帧差距只有 0.020 之后定的，这次没有改。",
            "",
            f"- 编码了 {summary['encoded_views']} 张画面。至少走出 1 米的转弯 {group['count']} 条。",
            f"- 这些转弯节点用的画面数中位数是 {group['median_end_views']:.1f} 张，"
            f"出发节点是 {group['median_start_views']:.1f} 张。",
            f"- 下一地点的相似度中位数 {group['median_cosine_start_end']:.3f}，"
            f"另一个地点 {group['median_cosine_start_other']:.3f}。",
            f"- 差距中位数 {group['median_delta']:.3f}。"
            f"{verdict['count']} 条里，下一地点更近的有 {verdict['count_closer']} 条，"
            f"差距达到 0.10 的有 {verdict['count_reaching_margin']} 条。",
            f"- 同一批样本如果仍只用一张，差距中位数是 {single['median_delta']:.3f}。",
            "",
            f"判定：`{verdict['call']}`。",
            closing,
            "没走的出口没有参加比较。",
            "",
        ]
    )


if __name__ == "__main__":
    main()
