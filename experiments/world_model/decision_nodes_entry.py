"""Mark decision nodes on saved views. Does not collect, train, or start Habitat."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.world_model.decision_nodes import (
    DEPARTURE_M,
    MIN_OPEN_DEG,
    MIN_WALL_DEG,
    OPEN_M,
    REACH_MARGIN_M,
    TAKEN_MATCH_DEG,
    WALL_M,
    WIDTH_RATIO,
    cluster_decision_frames,
    departure_bearing,
    exit_contrast,
    matched_exit,
    openings_from_depth,
)
from experiments.world_model.ego_warp import depth_to_meters
from experiments.world_model.ego_warp_entry import index_observations, load_checked
from experiments.world_model.event_spans import SAME_PLACE_M
from experiments.world_model.event_spans_entry import SPLITS, _poses

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output/world_model_diagnostics/decision_nodes_20261103_01"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    _write_json(OUT / "protocol.json", _protocol())
    nodes = []
    counts = {"frames": 0, "depth_missing": 0, "decision_frames": 0}
    for split, root in SPLITS.items():
        observations = index_observations(root)
        for episode_uid, poses in _poses(root).items():
            marked, frame_counts = _episode_nodes(split, root, episode_uid, poses, observations)
            nodes.extend(marked)
            for key, value in frame_counts.items():
                counts[key] += value
    _write_jsonl(OUT / "decision_nodes.jsonl", nodes)
    summary = _summary(nodes, counts)
    _write_json(OUT / "results.json", summary)
    (OUT / "report.md").write_text(_report(summary), encoding="utf-8")


def _protocol() -> dict:
    return {
        "question": "Where does one saved view show at least two walkable openings?",
        "node": "one place, merged within the same-place radius, represented by its clearest view",
        "opening": f"horizon sight of at least {OPEN_M} m",
        "separator": f"a near surface of at most {WALL_M} m spanning at least {MIN_WALL_DEG} degrees",
        "minimum_opening_deg": MIN_OPEN_DEG,
        "same_place_m": SAME_PLACE_M,
        "width_ratio": WIDTH_RATIO,
        "reach_margin_m": REACH_MARGIN_M,
        "taken_exit": (
            f"the opening whose bearing is within {TAKEN_MATCH_DEG} degrees of the first "
            f"{DEPARTURE_M} m departure"
        ),
        "corner_shape_labeled": False,
        "slope_labeled": False,
        "depth_used_only_to_mark_openings": True,
        "navigation_connected": False,
        "formal_cache_written": False,
        "training_started": False,
        "collection_started": False,
        "habitat_started": False,
    }


def _episode_nodes(split: str, root: Path, episode_uid: str, poses: list[dict], observations: dict) -> tuple[list[dict], dict]:
    frames = []
    counts = {"frames": 0, "depth_missing": 0, "decision_frames": 0}
    pose_index = {pose["observation_id"]: index for index, pose in enumerate(poses)}
    for pose in poses:
        counts["frames"] += 1
        observation = observations[pose["observation_id"]]
        if observation["depth_missing"]:
            counts["depth_missing"] += 1
            continue
        depth_m = depth_to_meters(load_checked(root, observation["depth"]))
        openings = openings_from_depth(depth_m)
        if len(openings) < 2:
            continue
        counts["decision_frames"] += 1
        frames.append(
            {
                **pose,
                "step_index": int(observation["step_index"]),
                "openings": openings,
            }
        )
    nodes = []
    for cluster in cluster_decision_frames(frames):
        representative = cluster["representative"]
        openings = representative["openings"]
        contrast = exit_contrast(openings)
        bearing = departure_bearing(poses, pose_index[representative["observation_id"]])
        taken = matched_exit(openings, bearing)
        nodes.append(
            {
                "split": split,
                "scene": representative["scene"],
                "episode_uid": episode_uid,
                "observation_id": representative["observation_id"],
                "step_index": representative["step_index"],
                "position": [float(part) for part in representative["position"]],
                "yaw": float(representative["yaw"]),
                "member_count": len(cluster["members"]),
                "member_observation_ids": [member["observation_id"] for member in cluster["members"]],
                "openings": openings,
                "wider_exit_id": contrast["wider_exit_id"],
                "longer_looking_exit_id": contrast["longer_looking_exit_id"],
                "departure_bearing_left_rad": bearing,
                "taken_exit_id": taken,
                "corner_shape_labeled": False,
                "slope_labeled": False,
            }
        )
    return nodes, counts


def _summary(nodes: list[dict], counts: dict) -> dict:
    two = [node for node in nodes if len(node["openings"]) == 2]
    taken = [node for node in nodes if node["taken_exit_id"] is not None]
    seen_not_walked = sum(len(node["openings"]) - 1 for node in taken)
    example = next((node for node in two if node["taken_exit_id"] is not None), None)
    return {
        **counts,
        "node_count": len(nodes),
        "nodes_with_two_openings": len(two),
        "nodes_with_more_than_two": len(nodes) - len(two),
        "nodes_with_taken_exit": len(taken),
        "exits_seen_not_walked": seen_not_walked,
        "nodes_with_wider_exit": sum(node["wider_exit_id"] is not None for node in nodes),
        "nodes_with_longer_looking_exit": sum(node["longer_looking_exit_id"] is not None for node in nodes),
        "example": _example(example) if example is not None else None,
        "corner_shape_labeled": False,
        "slope_labeled": False,
        "training_started": False,
        "collection_started": False,
        "habitat_started": False,
        "navigation_connected": False,
    }


def _example(node: dict) -> dict:
    return {
        "scene": node["scene"],
        "step_index": node["step_index"],
        "openings": [
            {
                "exit_id": opening["exit_id"],
                "side": opening["side"],
                "visible_reach_m": opening["visible_reach_m"],
                "apparent_width_m": opening["apparent_width_m"],
                "taken": opening["exit_id"] == node["taken_exit_id"],
            }
            for opening in node["openings"]
        ],
        "wider_exit_id": node["wider_exit_id"],
        "longer_looking_exit_id": node["longer_looking_exit_id"],
    }


def _report(summary: dict) -> str:
    example = summary["example"]
    example_line = "没有同时看清两条开口、又能对上走开方向的例子。"
    if example is not None:
        sides = "；".join(
            f"{_side_name(item['side'])}能看进去 {item['visible_reach_m']:.1f} 米，"
            f"这段看起来宽 {item['apparent_width_m']:.1f} 米"
            for item in example["openings"]
        )
        taken = next(item for item in example["openings"] if item["taken"])
        example_line = (
            f"例如场景 {example['scene']} 第 {example['step_index']} 步：{sides}。"
            f"这次走的是{_side_name(taken['side'])}。"
        )
    return "\n".join(
        [
            "# 路口节点",
            "",
            "一个节点标在同一张画面里能看见至少两个开口的地方。",
            "开口是相机正前方那一条水平线上，能看进去至少 3 米的一段。",
            "两段中间要隔着 2 米以内的墙，墙至少 8 度，每个开口至少 12 度。",
            "只有一条走廊时不标。同一次行走里，0.75 米以内反复看见的合成一个节点。",
            "另一次行走即使经过附近，也另外标，不当成同一次路口的两条出口。",
            "",
            "标记只用已经录下的深度来找开口，深度不进入导航。",
            "圆角和直角、上坡和下坡，单张深度分不出来，这次没有标。",
            "深度也分不出玻璃窗和真能走的路，所以开口表示“这边能看很远”。",
            "",
            f"- 看过 {summary['frames']} 张画面，缺深度 {summary['depth_missing']} 张。",
            f"- 能看见至少两个开口的画面 {summary['decision_frames']} 张，合成节点 {summary['node_count']} 个。",
            f"- 正好两个开口的节点 {summary['nodes_with_two_openings']} 个，"
            f"多于两个的 {summary['nodes_with_more_than_two']} 个。",
            f"- 能对上实际走开方向的节点 {summary['nodes_with_taken_exit']} 个，"
            f"对不上的 {summary['node_count'] - summary['nodes_with_taken_exit']} 个。",
            f"- 在对得上的节点里，看见了但这次没有走的出口 {summary['exits_seen_not_walked']} 条。",
            f"- 两个开口里，宽窄差得上的节点 {summary['nodes_with_wider_exit']} 个，"
            f"看得远近差得上的 {summary['nodes_with_longer_looking_exit']} 个。",
            "",
            example_line,
            "没出现在画面里的出口不会被标出来。",
            "",
        ]
    )


def _side_name(side: str) -> str:
    return {"left": "左边", "right": "右边", "center": "正前方"}[side]


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
