#!/usr/bin/env python3
"""Pair the official run with two pitch/roll jitter runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "output" / "r2r_official" / "results.jsonl"
ALLOW = Path(__file__).resolve().parent / "pitch_roll_allow.json"
RUNS = [
    ("baseline", None),
    ("jitter_seed11", ROOT / "output" / "r2r_camera_jitter" / "seed11" / "results.jsonl"),
    ("jitter_seed22", ROOT / "output" / "r2r_camera_jitter" / "seed22" / "results.jsonl"),
]
REST = ROOT / "output" / "r2r_camera_jitter" / "seed22_rest" / "results.jsonl"
OUT_JSON = ROOT / "output" / "r2r_camera_jitter" / "comparison.json"
OUT_MD = Path(__file__).resolve().parent / "pitch_roll_comparison.md"


def scene_of(scene_id: str) -> str:
    return os.path.basename(os.path.dirname(str(scene_id).rstrip("/")))


def success_of(row: dict) -> bool:
    value = row.get("success")
    if isinstance(value, bool):
        return value
    try:
        return float(value) >= 0.5
    except (TypeError, ValueError):
        return False


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    found = {}
    for row in rows:
        if "scene" in row and "episode_id" in row and "scene_id" not in row:
            key = (row["scene"], str(row["episode_id"]))
        else:
            key = (scene_of(row.get("scene_id", "")), str(row.get("habitat_episode_id", "")))
        found[key] = row
    return found


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    allow = json.loads(ALLOW.read_text())
    keys = [(row["scene"], str(row["episode_id"])) for row in allow]
    official = index_rows(load_jsonl(OFFICIAL))
    seed11 = index_rows(load_jsonl(RUNS[1][1]))
    seed22 = index_rows(load_jsonl(RUNS[2][1]) + load_jsonl(REST))
    tables = {"baseline": official, "jitter_seed11": seed11, "jitter_seed22": seed22}

    missing = {
        name: [f"{scene}/{episode}" for scene, episode in keys if (scene, episode) not in table]
        for name, table in tables.items()
    }
    paired = [key for key in keys if all(key in table for table in tables.values())]
    flags = {key: {name: success_of(tables[name][key]) for name in tables} for key in paired}
    disagree = [key for key, item in flags.items() if len(set(item.values())) > 1]
    union = [key for key, item in flags.items() if any(item.values())]
    patterns = {}
    for key, item in flags.items():
        label = "".join("S" if item[name] else "F" for name in ("baseline", "jitter_seed11", "jitter_seed22"))
        patterns[label] = patterns.get(label, 0) + 1

    summary_runs = {}
    for name, table in tables.items():
        rows = [table[key] for key in paired]
        successes = [success_of(row) for row in rows]
        summary_runs[name] = {
            "n": len(rows),
            "sr": mean([1.0 if item else 0.0 for item in successes]),
            "spl": mean([v for v in (finite(row.get("spl")) for row in rows) if v is not None]),
            "ndtw": mean([v for v in (finite(row.get("ndtw")) for row in rows) if v is not None]),
            "ne": mean([v for v in (finite(row.get("final_distance")) for row in rows) if v is not None]),
        }

    baseline_to_fail = sum(
        1 for item in flags.values() if item["baseline"] and not item["jitter_seed11"] and not item["jitter_seed22"]
    )
    fail_to_success = sum(
        1 for item in flags.values() if (not item["baseline"]) and (item["jitter_seed11"] or item["jitter_seed22"])
    )
    report = {
        "n_allow": len(keys),
        "n_paired": len(paired),
        "missing": missing,
        "runs": summary_runs,
        "disagree_n": len(disagree),
        "disagree_rate": None if not paired else len(disagree) / len(paired),
        "union_success_rate": None if not paired else len(union) / len(paired),
        "patterns_baseline_seed11_seed22": patterns,
        "baseline_success_both_jitter_fail": baseline_to_fail,
        "baseline_fail_either_jitter_success": fail_to_success,
        "protocol": "Each step, camera pitch and roll are independent Uniform(-1, 1) degrees. Body pose is unchanged. Official vlnce_r2r.yaml, checkpoints/LightNav-0.",
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    def pct(value: float | None) -> str:
        return "—" if value is None else f"{100 * value:.1f}%"

    def num(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    lines = [
        "# 俯仰和横滚抖动对照",
        "",
        "每一步把相机俯仰角和横滚角各自抽成 −1° 到 +1°，机身位置和朝向不动。",
        "配置是官方 `vlnce_r2r.yaml`，模型是 `checkpoints/LightNav-0`。",
        "样本是官方结果里抽出的 48 条，不是全部 1839 条。",
        "",
        "| 运行 | 条数 | SR | SPL | nDTW | NE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary_runs.items():
        lines.append(
            f"| {name} | {item['n']} | {pct(item['sr'])} | {num(item['spl'])} | {num(item['ndtw'])} | {num(item['ne'])} |"
        )
    lines += [
        "",
        f"三次结果不完全相同的回合：{len(disagree)} / {len(paired)}（{pct(report['disagree_rate'])}）。",
        f"三次里至少成功一次：{pct(report['union_success_rate'])}。",
        f"基线成功、两次抖动都失败：{baseline_to_fail}。基线失败、任一次抖动成功：{fail_to_success}。",
        "",
        "成败排列是基线、seed11、seed22。S 成功，F 失败。",
        "",
        "| 排列 | 条数 |",
        "|---|---:|",
    ]
    for label, count in sorted(patterns.items()):
        lines.append(f"| {label} | {count} |")
    if any(missing.values()):
        lines += ["", "缺的回合：", ""]
        for name, items in missing.items():
            if items:
                lines.append(f"- {name}: {', '.join(items)}")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "missing"}, indent=2))


if __name__ == "__main__":
    main()
