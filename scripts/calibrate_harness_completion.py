"""Replay labelled actual RGB cases through the task-aware completion verifier."""

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.jsonutil import sha256_file  # noqa: E402
from experiments.world_model.skill_contracts import GoalContract  # noqa: E402
from experiments.world_model.stateful_controller import save_json  # noqa: E402
from experiments.world_model.task_completion import QwenTaskAwareProgressVerifier  # noqa: E402
from experiments.world_model.task_progress import TaskProgress, TaskStage  # noqa: E402
from experiments.world_model.visual_harness import LocalQwenVision, ObservationImages  # noqa: E402


def set_case_depth(verifier, case):
    binding = case.get("current_depth")
    if binding is None:
        return False
    if binding["reference"] != case["current_reference"]:
        raise ValueError("depth must belong to the current actual frame")
    path = Path(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("actual depth binding changed")
    scale = float(binding["metres_per_depth_unit"])
    depth = np.load(path, allow_pickle=False)
    if (not math.isfinite(scale) or not 0 < scale <= 100 or depth.squeeze().ndim != 2 or
            not np.issubdtype(depth.dtype, np.number) or np.iscomplexobj(depth)):
        raise ValueError("invalid actual depth or scale")
    height, width = depth.squeeze().shape
    if (width, height) != verifier.images.get(binding["reference"]).size:
        raise ValueError("actual RGB and depth image dimensions must match")
    verifier.set_depth(depth, binding["reference"], scale)
    return True


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("visual calibration requires the actual CUDA model")
    torch.set_num_threads(2)
    cases = json.loads(args.cases.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__).resolve(), REPO / "experiments/world_model/task_completion.py",
                    REPO / "experiments/world_model/skill_contracts.py", REPO / "experiments/world_model/task_progress.py"]
    depth_files = [Path(c["current_depth"]["path"]) for c in cases if "current_depth" in c]
    save_json(args.output / "input_manifest.json", {str(p.resolve()): sha256_file(p) for p in [args.cases, *source_files, *depth_files]})
    for p in source_files:
        target = args.output / "source_snapshot" / p.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    vision = LocalQwenVision(args.model)
    rows = []
    for case in cases:
        images = ObservationImages()
        ordered = sorted(case["observations"], key=lambda r: r["step"])
        if any(r["step"] > case["step"] for r in ordered):
            raise ValueError("calibration must not reveal future frames")
        for row in ordered:
            images.register_file(row["reference"], row["path"], row["sha256"])
        stages = [TaskStage(**row) for row in (case["stages"] if "stages" in case else [case["stage"]])]
        tracker = TaskProgress(episode_id=case["case_id"], instruction=case["instruction"], stages=stages,
            initial_observation=ordered[0]["reference"], initial_step=ordered[0]["step"],
            require_current_target=GoalContract.from_instruction(stages[-1].instruction).require_current_target)
        for row in ordered[1:]:
            tracker.observe(row["reference"], row["step"])
        tracker.anchor = case["stage_anchor"]
        if tracker.anchor not in tracker.observations:
            raise ValueError("stage anchor lacks an actual observation")
        tracker.executions = {r["step"]: r for r in case["recent_actual_commands"]}
        verifier = QwenTaskAwareProgressVerifier(vision, images)
        verifier.set_execution_context({"actual_motion_source": "not recorded in this historical replay",
                                        "reference_goal_distance_not_available": True})
        set_case_depth(verifier, case)
        try:
            evidence = verifier(tracker, case["current_reference"])
            matched = evidence.verdict in case["accepted_verdicts"]
            if "expected_active_completed" in case:
                matched = matched and evidence.completed == case["expected_active_completed"]
            result = {"case_id": case["case_id"], "evidence": asdict(evidence), "trace": verifier.last_trace,
                "expected_verdicts_evaluation_only": case["accepted_verdicts"],
                "expected_active_completed_evaluation_only": case.get("expected_active_completed"),
                "matched_expected": matched,
                "two_view_stop_acceptance_tested": False}
        except (KeyError, ValueError, TypeError) as exc:
            result = {"case_id": case["case_id"], "error": str(exc), "trace": verifier.last_trace, "matched_expected": False}
        rows.append(result)
        save_json(args.output / "results.json", rows)
        print(json.dumps({"case": case["case_id"], "verdict": result.get("evidence", {}).get("verdict"),
                          "matched_expected": result["matched_expected"], "error": result.get("error")}), flush=True)
    save_json(args.output / "visual_calls.json", vision.calls)
    save_json(args.output / "complete.json", {"cases": len(rows), "matched": sum(r["matched_expected"] for r in rows),
        "selected_development_cases_not_blind_accuracy": True, "parameters_updated": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=REPO / "checkpoints/world_model_planner/Qwen3-VL-4B-Instruct")
    run(parser.parse_args())
