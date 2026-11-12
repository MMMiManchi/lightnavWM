"""Read saved poses and write node-to-node spans. Does not collect or encode."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from experiments.world_model.event_spans import (
    DISTINCT_EXIT_RAD,
    SAME_PLACE_M,
    STRAIGHT_STEP_RAD,
    TURN_RAD,
    WOBBLE_RESET_M,
    places_with_two_exits,
    recollection_verdict,
    segment_episode,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output/world_model_diagnostics/event_spans_20261103_01"
SPLITS = {
    "formal_dev_20261028_01": REPO / "output/world_model_data/formal_dev_20261028_01",
    "formal_fit_20261028_04": REPO / "output/world_model_data/formal_fit_20261028_04",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = {
        "question": "Can saved trajectories supply a node and the next node without a new collection?",
        "turn_rad": TURN_RAD,
        "straight_step_rad": STRAIGHT_STEP_RAD,
        "wobble_reset_m": WOBBLE_RESET_M,
        "same_place_m": SAME_PLACE_M,
        "distinct_exit_rad": DISTINCT_EXIT_RAD,
        "doors_and_forks_labeled": False,
        "control_step_is_the_target": False,
        "collection_started": False,
        "habitat_started": False,
        "training_started": False,
    }
    _write_json(OUT / "protocol.json", protocol)
    spans = []
    for split, root in SPLITS.items():
        spans.extend(_spans_for_split(split, root))
    with (OUT / "event_spans.jsonl").open("w") as handle:
        for span in spans:
            handle.write(json.dumps(span, ensure_ascii=False) + "\n")
    places = places_with_two_exits(spans)
    by_kind: dict[str, int] = defaultdict(int)
    for span in spans:
        by_kind[span["end_kind"]] += 1
    verdict = recollection_verdict(span_count=len(spans), place_count=len(places))
    results = {
        "span_count": len(spans),
        "by_end_kind": dict(by_kind),
        "by_split": {
            split: sum(1 for span in spans if span["split"] == split) for split in SPLITS
        },
        "median_path_m": _median([span["path_m"] for span in spans]),
        "median_recorded_control_steps": _median(
            [span["recorded_control_steps"] for span in spans]
        ),
        "places_with_two_exits": places,
        "verdict": verdict,
    }
    _write_json(OUT / "results.json", results)
    (OUT / "report.md").write_text(_report(results), encoding="utf-8")


def _spans_for_split(split: str, root: Path) -> list[dict]:
    poses_by_episode = _poses(root)
    spans = []
    for episode_uid, poses in poses_by_episode.items():
        for span in segment_episode(poses):
            span["split"] = split
            span["episode_uid"] = episode_uid
            spans.append(span)
    return spans


def _poses(root: Path) -> dict[str, list[dict]]:
    labels = {}
    with (root / "privileged_labels.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            pose = row["pose"]
            labels[row["label_id"]] = {
                "position": tuple(pose["world_position_xyz_m"]),
                "yaw": float(pose["derived_yaw_rad"]),
            }
    grouped: dict[str, list[dict]] = defaultdict(list)
    with (root / "transitions.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            grouped[row["episode_uid"]].append(row)
    episodes = {}
    for episode_uid, rows in grouped.items():
        rows.sort(key=lambda row: row["t"])
        scene = episode_uid.split("|")[2]
        sequence = [
            {
                "scene": scene,
                "episode_uid": episode_uid,
                "observation_id": rows[0]["obs_before_id"],
                **labels[rows[0]["pose_before_id"]],
            }
        ]
        for row in rows:
            sequence.append(
                {
                    "scene": scene,
                    "episode_uid": episode_uid,
                    "observation_id": row["obs_after_id"],
                    **labels[row["pose_after_id"]],
                }
            )
        episodes[episode_uid] = sequence
    return episodes


def _write_json(path: Path, value: dict) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _report(results: dict) -> str:
    places = results["places_with_two_exits"]
    non_start = [place for place in places if not place["includes_only_episode_starts"]]
    return "\n".join(
        [
            "# 节点到下一节点",
            "",
            "样本是一次已完成的转弯，或走到回合结束。控制周期留在样本内部，不是预测目标。",
            "日志里没有门口和岔路标签，直行穿过的门不会被单独切开。",
            "",
            f"- 样本 {results['span_count']} 条，其中转弯结束 {results['by_end_kind'].get('turn', 0)} 条，"
            f"回合结束 {results['by_end_kind'].get('episode_end', 0)} 条。",
            f"- 路径长度中位数 {results['median_path_m']:.2f} 米。",
            f"- 记录用的控制步中位数 {results['median_recorded_control_steps']:.0f}。",
            f"- 同一地点有两个不同离开方向的地点 {len(places)} 个，"
            f"其中不是回合起点的 {len(non_start)} 个。",
            "",
            f"判定：`{results['verdict']['call']}`。",
            "",
            f"这 {len(places)} 个地点是不同行走在 0.75 米内、离开朝向不同。",
            "它们不是同一次决策里看过的两条岔路，没走的出口仍然没有画面。",
            "",
        ]
    )


if __name__ == "__main__":
    main()
