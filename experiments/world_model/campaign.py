"""Append-only campaign manifests and cross-run collection coverage reports."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.world_model.baselines import action_kind, executed_twist
from experiments.world_model.jsonutil import dumps, sha256_file
from experiments.world_model.latent_data import scene_partition
from experiments.world_model.validate import validate_run

CAMPAIGN_SCHEMA = "wm-campaign/0.1"
_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_campaign_target(manifest_path: Path, campaign_id: str, run_dir: Path) -> None:
    """Read-only preflight before a collector creates its run directory."""
    if not _CAMPAIGN_ID.fullmatch(campaign_id):
        raise ValueError("campaign id must be 1-128 ASCII letters, digits, dot, underscore or hyphen")
    manifest_path = Path(manifest_path).resolve()
    run_dir = Path(run_dir).resolve()
    if manifest_path.is_relative_to(run_dir):
        raise ValueError("campaign manifest must be outside the run directory")
    if manifest_path.exists():
        campaign = _read_json(manifest_path)
        if campaign.get("schema_version") != CAMPAIGN_SCHEMA or campaign.get("campaign_id") != campaign_id:
            raise ValueError("campaign manifest schema or id mismatch")
        if campaign.get("scope") != "train_fit" or campaign.get("source_split") != "train":
            raise ValueError("campaign manifest is not train_fit")
        if any(item.get("run_dir") == str(run_dir) for item in campaign.get("runs") or []):
            raise ValueError(f"run is already registered: {run_dir}")


def register_campaign_run(manifest_path: Path, campaign_id: str, run_dir: Path) -> None:
    """Register one validated immutable run without replacing prior campaign entries."""
    validate_campaign_target(manifest_path, campaign_id, run_dir)
    manifest_path = Path(manifest_path).resolve()
    run_dir = Path(run_dir).resolve()
    run_manifest = _read_json(run_dir / "manifest.json")
    report = _read_json(run_dir / "validation_report.json")
    if run_manifest.get("campaign_id") != campaign_id:
        raise ValueError("run campaign_id does not match the requested campaign")
    if (run_manifest.get("source") or {}).get("split") != "train":
        raise ValueError("only train split runs can enter a campaign manifest")
    if run_manifest.get("diagnostic_only") is True or report.get("hard_error_count") != 0:
        raise ValueError("only non-diagnostic, hard-error-free runs can enter a campaign manifest")
    episode_rows = [
        json.loads(line)
        for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(
        row.get("source_split") != "train"
        or scene_partition(str(row.get("scene"))) != "train_fit"
        for row in episode_rows
    ):
        raise ValueError("only train_fit episodes can enter a campaign manifest")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            campaign = _read_json(manifest_path)
            if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
                raise ValueError("campaign manifest schema mismatch")
            if campaign.get("campaign_id") != campaign_id or campaign.get("scope") != "train_fit":
                raise ValueError("campaign identity or scope mismatch")
        else:
            campaign = {
                "schema_version": CAMPAIGN_SCHEMA,
                "campaign_id": campaign_id,
                "scope": "train_fit",
                "source_split": "train",
                "created_at": run_manifest.get("created_at"),
                "runs": [],
            }
        run_path = str(run_dir)
        if any(item.get("run_dir") == run_path for item in campaign["runs"]):
            raise ValueError(f"run is already registered: {run_path}")
        campaign["runs"].append(
            {
                "run_dir": run_path,
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "validation_report_sha256": sha256_file(run_dir / "validation_report.json"),
            }
        )
        fd, temporary = tempfile.mkstemp(prefix=f".{manifest_path.name}.", dir=manifest_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(dumps(campaign) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest_path)
            directory_fd = os.open(manifest_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def aggregate_campaign(manifest_path: Path) -> dict[str, Any]:
    """Revalidate all registered runs and report raw train_fit coverage only."""
    campaign = _read_json(Path(manifest_path))
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA or campaign.get("scope") != "train_fit":
        raise ValueError("unsupported campaign manifest or scope")
    if campaign.get("source_split") != "train" or not _CAMPAIGN_ID.fullmatch(str(campaign.get("campaign_id", ""))):
        raise ValueError("campaign source or id is invalid")
    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    seen_episode_uids: set[str] = set()
    runs = campaign.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("campaign manifest must list at least one run")
    run_reports = []
    for item in runs:
        run_dir = Path(item["run_dir"]).resolve()
        if sha256_file(run_dir / "manifest.json") != item.get("manifest_sha256"):
            raise ValueError(f"run manifest changed after registration: {run_dir}")
        if sha256_file(run_dir / "validation_report.json") != item.get("validation_report_sha256"):
            raise ValueError(f"run validation report changed after registration: {run_dir}")
        run_manifest = _read_json(run_dir / "manifest.json")
        if run_manifest.get("campaign_id") != campaign["campaign_id"]:
            raise ValueError(f"run belongs to another campaign: {run_dir}")
        if run_manifest.get("diagnostic_only") is True or (run_manifest.get("source") or {}).get("split") != "train":
            raise ValueError(f"run is not eligible train data: {run_dir}")
        report = validate_run(run_dir)
        if report.get("hard_error_count") != 0:
            raise ValueError(f"run revalidation failed: {run_dir}")
        run_episodes = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
        run_transitions = [json.loads(line) for line in (run_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for episode in run_episodes:
            uid = episode.get("episode_uid")
            if uid in seen_episode_uids:
                raise ValueError(f"duplicate episode across campaign runs: {uid}")
            seen_episode_uids.add(uid)
            if episode.get("source_split") != "train" or scene_partition(str(episode.get("scene"))) != "train_fit":
                raise ValueError(f"campaign contains a non-train_fit episode: {uid}")
        episodes.extend(run_episodes)
        transitions.extend(run_transitions)
        run_reports.append({"run_dir": str(run_dir), "hard_error_count": 0, "episodes": len(run_episodes), "transitions": len(run_transitions)})

    scene_counts: Counter[str] = Counter(str(row["scene"]) for row in episodes)
    routes = {str(row["route_group_id"]) for row in episodes if row.get("route_group_id")}
    action_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    budget_episodes: set[str] = set()
    for row in transitions:
        outcome = row.get("outcome") or {}
        primary = str(outcome.get("primary_end_source") or "unknown")
        primary_counts[primary] += 1
        event_counts.update(str(value) for value in (outcome.get("end_sources") or []))
        if primary == "runner_budget":
            budget_episodes.add(str(row.get("episode_uid")))
        if (row.get("quality") or {}).get("status") == "valid":
            action_counts[action_kind(executed_twist(row))] += 1
    non_budget_completed = sum(
        1 for row in episodes
        if (row.get("completion") or {}).get("state") == "completed"
        and str(row.get("episode_uid")) not in budget_episodes
    )
    coverage_blockers = []
    if len(scene_counts) < 8:
        coverage_blockers.append(f"train_fit scenes {len(scene_counts)}/8")
    if len(routes) < 20:
        coverage_blockers.append(f"train_fit route groups {len(routes)}/20")
    if any(count < 2 for count in scene_counts.values()):
        coverage_blockers.append("at least one sampled scene has fewer than 2 episodes")
    for name in ("stop_threshold", "turning", "mostly_forward"):
        if action_counts.get(name, 0) < 20:
            coverage_blockers.append(f"{name} valid actions {action_counts.get(name, 0)}/20")
    if non_budget_completed < 20:
        coverage_blockers.append(f"non-budget completed episodes {non_budget_completed}/20")
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "scope": "train_fit",
        "run_count": len(runs),
        "runs": run_reports,
        "raw_coverage": {
            "scenes": len(scene_counts),
            "route_groups": len(routes),
            "episodes_per_scene": dict(sorted(scene_counts.items())),
            "valid_action_kind_counts": {name: action_counts.get(name, 0) for name in ("stop_threshold", "turning", "mostly_forward")},
            "primary_end_source_counts": dict(sorted(primary_counts.items())),
            "end_source_event_counts": dict(sorted(event_counts.items())),
            "episode_count": len(episodes),
            "non_budget_completed_episodes": non_budget_completed,
        },
        "raw_coverage_floor_met": not coverage_blockers,
        "raw_coverage_blockers": coverage_blockers,
        "quota_eligible_transition_count": 0,
        "quota_blockers": ["paired latent features are not authorized or available"],
        "training_authorized": False,
        "training_admitted": False,
        "note": "This report aggregates collection coverage only; it does not admit training or evaluation.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Revalidate and summarize an incremental train_fit campaign")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(aggregate_campaign(Path(args.manifest)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
