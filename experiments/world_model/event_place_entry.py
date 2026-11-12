"""Encode event endpoints and compare the start view with the next node.

Uses the audited SigLIP vision tower. Writes a diagnostic score, not the formal
feature cache. Does not collect, train, or start Habitat.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.world_model.cache_writer import verify_bound_files
from experiments.world_model.ego_warp_entry import index_observations, load_checked
from experiments.world_model.event_place import LONG_PATH_M, cosine, other_end_ids, place_verdict
from experiments.world_model.feature_cache import (
    AUDITED_PREPROCESSOR,
    OUTPUT_SHAPES,
    SIGLIP_REVISION,
    SIGLIP_WEIGHT_SHA256,
)

REPO = Path(__file__).resolve().parents[2]
SPANS = REPO / "output/world_model_diagnostics/event_spans_20261103_01/event_spans.jsonl"
OUT = REPO / "output/world_model_diagnostics/event_place_20261103_01"
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


def main() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {str(index) for index in range(2, 8)}:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to a single GPU from 2 to 7")
    OUT.mkdir(parents=True, exist_ok=False)
    _write_json(
        OUT / "protocol.json",
        {
            "question": "Is the next node visually closer to the start node than to another arrival?",
            "encoder": "SiglipVisionModel pooler_output",
            "output_shape": list(OUTPUT_SHAPES["pooler_output"]),
            "similarity": "cosine of the raw pooler vectors",
            "long_path_m": LONG_PATH_M,
            "verdict_group": "turn spans whose path is at least one meter",
            "other_place": "next span arrival in the same split, skipping this span's own views",
            "feature_layer_selected": None,
            "formal_cache_written": False,
            "training_started": False,
            "collection_started": False,
            "habitat_started": False,
            "branch_choice_tested": False,
        },
    )
    spans = [json.loads(line) for line in SPANS.read_text(encoding="utf-8").splitlines() if line]
    observations = {split: index_observations(run) for split, run in RUNS.items()}
    needed: dict[tuple[str, str], dict] = {}
    for span in spans:
        for key in ("start_observation_id", "end_observation_id"):
            needed[(span["split"], span[key])] = observations[span["split"]][span[key]]
    vectors = _encode(needed)
    np.savez_compressed(
        OUT / "endpoint_pooler.npz",
        ids=np.array(["|".join(key) for key in vectors], dtype=object),
        features=np.stack([vectors[key] for key in vectors]),
    )
    others = other_end_ids(spans)
    rows = []
    for span, other_id in zip(spans, others, strict=True):
        start = vectors[(span["split"], span["start_observation_id"])]
        end = vectors[(span["split"], span["end_observation_id"])]
        other = vectors[(span["split"], other_id)]
        rows.append(
            {
                **span,
                "other_observation_id": other_id,
                "cosine_start_end": cosine(start, end),
                "cosine_start_other": cosine(start, other),
            }
        )
    _write_jsonl(OUT / "place_scores.jsonl", rows)
    group = [
        row
        for row in rows
        if row["end_kind"] == "turn" and row["path_m"] >= LONG_PATH_M
    ]
    verdict = place_verdict(
        [row["cosine_start_end"] for row in group],
        [row["cosine_start_other"] for row in group],
    )
    summary = {
        "encoded_views": len(vectors),
        "span_count": len(rows),
        "verdict_count": len(group),
        "groups": {
            "all": _group_summary(rows),
            "turn": _group_summary([row for row in rows if row["end_kind"] == "turn"]),
            "turn_at_least_1m": _group_summary(group),
            "episode_end": _group_summary([row for row in rows if row["end_kind"] == "episode_end"]),
        },
        "verdict": verdict,
    }
    _write_json(OUT / "results.json", summary)
    (OUT / "report.md").write_text(_report(summary), encoding="utf-8")


def _encode(needed: dict[tuple[str, str], dict]) -> dict[tuple[str, str], np.ndarray]:
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel

    weight = SNAPSHOT / "model.safetensors"
    if verify_bound_files(weight)["weight_sha256"] != SIGLIP_WEIGHT_SHA256:
        raise RuntimeError("SigLIP weight hash does not match the audited contract")
    if torch.version.cuda != "12.8" or torch.__version__ != "2.10.0+cu128":
        raise RuntimeError(f"unapproved torch/CUDA runtime: {torch.__version__}")
    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    model = SiglipVisionModel.from_pretrained(
        SNAPSHOT, local_files_only=True, torch_dtype=torch.float32
    ).to("cuda:0").eval()
    keys = list(needed)
    images = [load_checked(RUNS[split], meta["rgb"]) for split, meta in ((key[0], needed[key]) for key in keys)]
    features = np.empty((len(keys), *OUTPUT_SHAPES["pooler_output"]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(images), BATCH):
            chunk = images[start : start + BATCH]
            pixels = processor(images=chunk, return_tensors="pt")["pixel_values"].to("cuda:0")
            output = model(pixel_values=pixels).pooler_output.float().cpu().numpy()
            if output.shape != (len(chunk), *OUTPUT_SHAPES["pooler_output"]) or not np.isfinite(output).all():
                raise RuntimeError(f"unexpected pooler output {output.shape}")
            features[start : start + len(chunk)] = output
    return {key: features[index] for index, key in enumerate(keys)}


def _group_summary(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "median_cosine_start_end": _median([row["cosine_start_end"] for row in rows]),
        "median_cosine_start_other": _median([row["cosine_start_other"] for row in rows]),
        "median_delta": _median(
            [row["cosine_start_end"] - row["cosine_start_other"] for row in rows]
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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _report(summary: dict) -> str:
    verdict = summary["verdict"]
    group = summary["groups"]["turn_at_least_1m"]
    return "\n".join(
        [
            "# 下一节点是否还像起点",
            "",
            "比较的是冻结 SigLIP 视觉塔的 pooler 向量。没有训练，没有写入正式特征缓存。",
            "另一地点是同一划分里下一条样本的到达画面。",
            "",
            f"- 编码了 {summary['encoded_views']} 张节点画面，样本 {summary['span_count']} 条。",
            f"- 至少走出 1 米并且停在转弯的 {group['count']} 条：",
            f"  起点对到达的余弦中位数 {group['median_cosine_start_end']:.3f}，",
            f"  起点对另一地点的余弦中位数 {group['median_cosine_start_other']:.3f}，",
            f"  差值中位数 {group['median_delta']:.3f}。",
            "",
            f"判定：`{verdict['call']}`。",
            "这只说明已走路径的下一节点和起点像不像。没走的出口没有参加比较。",
            "",
        ]
    )


if __name__ == "__main__":
    main()
