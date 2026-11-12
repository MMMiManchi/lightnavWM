"""Connect admitted event prediction and real past memory to frozen LightNav.

Recorded inference uses a declared teacher direction as an interface probe.
Optional live execution retrieves environmental evidence without automatically
following a remembered direction. Without an explicit current-task planner,
LightNav retains the original episode instruction. This is not a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.event_runtime import (  # noqa: E402
    HarnessLightNavExecutor,
    LocalGoal,
    load_residual_predictor,
    predict_branch_with_memory,
)
from experiments.world_model.event_train import digest, read_rows, save  # noqa: E402
from experiments.world_model.feature_cache import AUDITED_PREPROCESSOR, SIGLIP_WEIGHT_SHA256  # noqa: E402
from experiments.world_model.latent_data import load_accepted_run_set, read_admitted_rgb  # noqa: E402
from experiments.world_model.online_input import online_info, online_observation  # noqa: E402
from experiments.world_model.scene_memory import PersistentSceneMemory  # noqa: E402
from experiments.world_model.task_planning import (  # noqa: E402
    EnvironmentEvidence,
    TaskGoalRouter,
    TaskPlanningContext,
)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def select_checkpoint(root):
    rows = [json.loads(p.read_text()) for p in root.glob("*/result.json")]
    rows = [r for r in rows if r["arm"]["use_branch"] and r["arm"]["use_memory"]]
    chosen = max(rows, key=lambda r: r["internal"]["scene_macro_cosine"] + 0.1*r["internal"]["recall_at_1"])
    arm = chosen["arm"]
    path = root / f"{arm['name']}_seed{arm['seed']}"
    control = chosen["baselines"]["direct_same_history_without_world_model"]
    initial = json.loads((path / "history.jsonl").read_text().splitlines()[0])["metrics"]
    def score(x):
        return x["scene_macro_cosine"] + 0.1*x["recall_at_1"]
    passed = (chosen["internally_selected_updates"] > 0 and score(chosen["internal"]) > score(initial) and
              chosen["development"]["scene_macro_cosine"] >= control["scene_macro_cosine"] and
              chosen["development"]["recall_at_1"] >= control["recall_at_1"])
    if not passed:
        raise RuntimeError("internal selection / same-history non-regression gate failed")
    return path / "model.pt", chosen


def seed_prior_memory(config, predictor, output, chosen_index=None):
    base = Path(config["base_run"])
    pairs = read_rows(base / "pairs.jsonl")
    arrays = np.load(base / "tensors.npz")
    order = [tuple(x) for x in json.loads((Path(config["run_dir"]) / "episode_order.json").read_text())]
    rank = {ep: i for i, ep in enumerate(order)}
    indices = np.load(Path(config["run_dir"]) / "memory_indices_1.npy")
    possible = [i for i, p in enumerate(pairs) if p["partition"] == "train_dev" and (indices[i] >= 0).any()]
    chosen = (min(possible, key=lambda i: (rank[(pairs[i]["source_run"], pairs[i]["episode_uid"])], pairs[i]["origin_index"]))
              if chosen_index is None else chosen_index)
    if chosen not in possible:
        raise ValueError("probe needs a dev sample with strictly prior-episode memory")
    query = pairs[chosen]
    before = rank[(query["source_run"], query["episode_uid"])]
    memory = PersistentSceneMemory(output / "past_experience.sqlite", contract_id=predictor.feature_contract_id)
    prior = sorted([i for i, p in enumerate(pairs) if p["partition"] == "train_dev" and p["scene"] == query["scene"] and
                    rank[(p["source_run"], p["episode_uid"])] < before],
                   key=lambda i: (rank[(pairs[i]["source_run"], pairs[i]["episode_uid"])], pairs[i]["origin_index"]))
    for i in prior:
        p = pairs[i]
        memory.record_observed(context=query["scene"], episode_id=p["episode_uid"], event_id=p["sample_id"],
                    source_observation=p["history_observation_ids"][-1], target_observation=p["target_observation_id"],
                    execution_ref=f"approved-recorded-execution:{p['sample_id']}", source_feature=arrays["history"][i, -1],
                    observed_future=arrays["target"][i], branch_direction=arrays["branch"][i], executed=True, arrived=True)
    return memory, chosen, query, arrays


def raw_features(base):
    result = {}
    for path in sorted(base.glob("features_*.npz")):
        a = np.load(path)
        result.update(zip(a["ids"].tolist(), a["features"], strict=True))
    return result


def source_admission(base, query_run):
    cfg = json.loads((base / "config.json").read_text())
    sources = []
    # These previously approved fit/dev runs independently meet the combined
    # admission floors. Only this dev run supplies RGB pixels to this smoke.
    # The public loader still performs every approval, pixel and split check.
    selected_runs = list(dict.fromkeys([cfg["source_runs"][0], query_run]))
    for run in selected_runs:
        folder = "motion_supplement_20261030_01" if "radius010" in run else "first_v1_20261029"
        sources.append((REPO / "output/world_model_data" / run,
                        REPO / "output/world_model_approvals" / folder / f"{run}.approval.json", Path(cfg["approval_key"])))
    return load_accepted_run_set(sources)


def calibrate_revisit_floor(root):
    """Use only internal fit-validation scenes, before any live observations."""
    config = json.loads((root / "config.json").read_text())
    base = Path(config["base_run"])
    pairs = read_rows(base / "pairs.jsonl")
    arrays = np.load(base / "tensors.npz")
    indices = np.load(root / "memory_indices_0.npy")
    scenes = set(json.loads((root / "split.json").read_text())["internal_validation_scenes"])
    observations = []
    for i, pair in enumerate(pairs):
        if pair["scene"] not in scenes or indices[i, 0] < 0:
            continue
        j = indices[i, 0]
        observations.append((float(arrays["history"][i, -1] @ arrays["history"][j, -1]),
                             pair["offline_labels"]["origin_place"] == pairs[j]["offline_labels"]["origin_place"]))
    grid = []
    for floor in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98):
        accepted = [correct for cosine, correct in observations if cosine >= floor]
        grid.append({"threshold": floor, "queries": len(accepted), "precision": float(np.mean(accepted)) if accepted else None})
    eligible = [row for row in grid if row["queries"] >= 20 and row["precision"] >= 0.95]
    if not eligible:
        raise ValueError("internal evidence does not support a visual revisit trigger")
    return {"selected_threshold": eligible[0]["threshold"], "grid": grid,
            "source": "8 internal fit-validation scenes; scene-local previous-episode sources",
            "labels": "offline v4 place labels only for calibration, not runtime",
            "limitation": "weak-label place agreement; heading and traversability not certified"}


def local_instruction(angle):
    if angle > math.radians(10):
        return "Take the visible passage on your left and move through the entrance."
    if angle < -math.radians(10):
        return "Take the visible passage on your right and move through the entrance."
    return "Continue through the visible passage ahead."


def build_base(args):
    from lightnav.habitat.runner import HabitatEvalConfig, build_velocity_policy
    from lightnav.inference import InferenceConfig, build_engine

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not fall back to CPU or another checkpoint")
    engine, bundle = build_engine(InferenceConfig(model_path=str(args.model_path), backend=args.backend,
                                                  gpu_memory_utilization=0.45, max_num_seqs=1, max_new_tokens=32),
                                  task_type="vlnce", max_new_tokens=32)
    cfg = HabitatEvalConfig(model_path=str(args.model_path), backend=args.backend, max_new_tokens=32)
    return engine, bundle, lambda info: build_velocity_policy(cfg, engine, bundle, info)


def live_smoke(args, config, predictor, bank, query, factory, output, calibration):
    from PIL import Image
    from transformers import SiglipImageProcessor, SiglipVisionModel

    from lightnav.habitat.remote_env import RemoteEnvClient

    snapshot = Path(config["encoder_snapshot"])
    if digest(snapshot / "model.safetensors") != SIGLIP_WEIGHT_SHA256:
        raise ValueError("audited visual encoder changed")
    encoder = SiglipVisionModel.from_pretrained(snapshot, local_files_only=True, dtype=torch.float32).cuda().eval()
    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    env = RemoteEnvClient(args.server, timeout_ms=60000, recv_retries=0)
    visible = bank.latest_sequence(query["scene"])
    steps, rounds, activated, recovered = [], [], False, False
    observation_turns = 0
    predictive = visual = None
    executor = None
    try:
        obs, info = env.reset()
        executor = HarnessLightNavExecutor(factory(online_info(info)))
        executor.reset(online_observation(obs))
        Image.fromarray(np.asarray(obs["rgb"])).save(output / "live_initial.png")
        expected_uid = query["episode_uid"].split("|")[-1]
        # Episode identity is checked for the experiment setup, never sent to
        # the learned policy or used as a node/adjacency lookup.
        served = info.get("episode_id")
        if served is not None and str(served).split("|")[-1] != expected_uid:
            raise ValueError(f"live server episode {served} does not match the declared smoke setup")
        episode_id = f"live-smoke:{query['episode_uid']}"
        router = TaskGoalRouter(executor, episode_id)
        if args.planner_model is not None:
            from experiments.world_model.predictive_harness import PredictiveTaskHarness
            from experiments.world_model.visual_harness import (
                LocalQwenVision,
                ObservationImages,
                QwenBranchProposer,
                QwenTaskPlanner,
                register_event_images,
            )
            images = ObservationImages()
            register_event_images(images, Path(config["base_run"]))
            visual = LocalQwenVision(args.planner_model, args.planner_device)
            proposer = QwenBranchProposer(visual, images)
            planner = QwenTaskPlanner(visual, images)
            predictive = PredictiveTaskHarness(executor=executor, predictor=predictor, bank=bank,
                         memory_context=query["scene"], episode_id=episode_id, images=images,
                         proposer=proposer, planner=planner, query_mode=args.query_mode)

        def planning_round(step, record):
            nonlocal activated, recovered, observation_turns
            folder = output / "planning_images"
            folder.mkdir(exist_ok=True)
            predictive.images.get(predictive.observations[-1]).save(folder / f"step_{step:03d}.png")
            try:
                planned = predictive.plan()
                rounds.append({"step": step, **planned.trace})
                record["planning_round"] = len(rounds)-1
                if planned.goal is not None:
                    activated, recovered = True, False
                    record.update(local_goal_active=True, local_instruction=planned.goal.instruction,
                                  retrieved_past_sequences=list(planned.goal.evidence_sequences),
                                  native_history_frames_preserved=len(executor.policy.agent._history_frame_ids))
                    predictive.report_execution(f"At step {step}, planner selected local instruction: {planned.goal.instruction}; execution and completion still unverified.")
                record.update(local_goal_active=executor.active_goal is not None,
                              recovered_global_instruction=recovered and executor.active_goal is None)
                if planned.action == "observe" and abs(planned.observation_turn_rad) > 1e-6:
                    if observation_turns < args.maximum_observation_turns:
                        from lightnav.velocity import first_waypoint_to_velocity_cmd
                        observation_turns += 1
                        executor.restore_global_goal()
                        record.update(active_observation=True, local_goal_active=False)
                        executor.last_tool_status = "observation_action_proposed"
                        return first_waypoint_to_velocity_cmd(np.array([0., 0., planned.observation_turn_rad]),
                            executor.policy.dt, executor.policy.lin_vel_range, executor.policy.ang_vel_range)
                    record["observation_budget_exhausted"] = True
                return None
            except (ValueError, KeyError, TypeError) as error:
                executor.restore_global_goal()
                record.update(planning_error=str(error), local_goal_active=False)
                rounds.append({"step": step, **predictive.last_trace, "error": str(error), "action": "continue_global"})
                predictive.report_execution(f"Planning tool rejected at step {step}: {error}; no goal completion inferred.")
                return None
            finally:
                save(output / "planning_trace.json", rounds)
                save(output / "visual_tool_calls.json", visual.calls)

        for step in range(args.live_steps):
            rgb = np.asarray(obs["rgb"])
            tensor = processor(images=[rgb], return_tensors="pt")["pixel_values"].cuda()
            with torch.inference_mode():
                vision = encoder(tensor)
            feature = vision.pooler_output.float().cpu().numpy()[0]
            canonical = predictor.normalize(torch.from_numpy(feature)[None])[0].numpy()
            now = f"live:{step}"
            revisits = bank.nearest_observed_sources(context=query["scene"], before_sequence=visible,
                            source_feature=canonical, exclude_episode=episode_id,
                            source_floor=calibration["selected_threshold"])
            record = {"step": step, "local_goal_active": executor.active_goal is not None,
                      "recovered_global_instruction": recovered,
                      "visual_revisit_candidates": [{"sequence": m.sequence, "cosine": m.source_similarity} for m in revisits]}
            record["environment_evidence_sequences"] = [m.sequence for m in revisits]
            action = None
            if predictive is not None:
                columns = vision.last_hidden_state.reshape(1, 27, 27, 1152).mean(1).float().cpu()
                predictive.observe(now, rgb, feature, columns)
                if step % args.planning_interval == 0:
                    action = planning_round(step, record)
            else:
                context = TaskPlanningContext(episode_id, executor.original_instruction, now, visible,
                                              EnvironmentEvidence(tuple(revisits)))
                router.consider(context)
                record["task_planning_status"] = router.last_status
            if action is not None:
                # Observation motion still gets exactly one real current frame.
                executor.policy.observe(online_observation(obs), online_info(info))
                details = {"rvq_codes": None}
            else:
                action, details = executor.propose_action(online_observation(obs), online_info(info))
            record.update(tool_status=executor.last_tool_status, rvq_codes=details.get("rvq_codes"), action=serial(action))
            if action is None and executor.last_tool_status == "stop_proposed":
                record["original_stop_proposal"] = serial({"source": "LightNav", "raw_text": details.get("raw_text"),
                                         "rvq_codes": details.get("rvq_codes"), "trajectory": details.get("predicted_traj")})
            if action is None and executor.active_goal is not None and executor.last_tool_status == "stop_proposed":
                # This smoke has no validated next-event arrival. Do not let a
                # short local instruction terminate the original route task.
                before = len(executor.policy.agent._history_frame_ids)
                record["original_stop_proposal"] = serial({"source": "LightNav", "raw_text": details.get("raw_text"),
                                         "rvq_codes": details.get("rvq_codes"), "trajectory": details.get("predicted_traj")})
                executor.restore_global_goal()
                recovered = True
                if predictive is not None:
                    predictive.report_execution(f"Local policy proposed STOP at step {step}; arrival unverified; original instruction restored.")
                    action = planning_round(step, record)
                if action is None:
                    action, details = executor.replan_observed_frame()
                else:
                    details = {"rvq_codes": None}
                after = len(executor.policy.agent._history_frame_ids)
                if before != after:
                    raise ValueError("replanning duplicated the already observed frame")
                record.update(local_goal_stop_rejected=True, local_goal_active=executor.active_goal is not None,
                              recovered_global_instruction=executor.active_goal is None, replan_history_frames=after,
                              tool_status=executor.last_tool_status, action=serial(action), rvq_codes=details.get("rvq_codes"))
            if action is None:
                steps.append(record)
                break
            old_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
            obs, _reward, terminated, truncated, info = env.step(action)
            record.update(environment_step_returned=True, rgb_changed=old_hash != hashlib.sha256(np.asarray(obs["rgb"]).tobytes()).hexdigest(),
                          terminated=terminated, truncated=truncated)
            record["evaluation_only"] = serial({k: info[k] for k in ["distance_to_goal", "success", "spl"] if k in info})
            steps.append(record)
            save(output / "live_trace.json", serial(steps))
            print(json.dumps({"live_step": step, "goal_ever_activated": activated,
                              "local_goal_active": record["local_goal_active"], "rgb_changed": record["rgb_changed"]}), flush=True)
            if terminated or truncated:
                break
        Image.fromarray(np.asarray(obs["rgb"])).save(output / "live_final.png")
        save(output / "live_final_evaluation.json", serial({k: info[k] for k in ["distance_to_goal", "success", "spl", "episode_id", "scene_id"] if k in info}))
    finally:
        env.close()
    save(output / "live_trace.json", serial(steps))
    save(output / "planning_trace.json", rounds)
    if visual is not None:
        save(output / "visual_tool_calls.json", visual.calls)
    return {"environment_connected": True, "steps_executed": sum(r.get("environment_step_returned", False) for r in steps),
            "local_goal_activated": activated, "executed_steps_after_activation": sum(r.get("environment_step_returned", False) and r["local_goal_active"] for r in steps),
            "visual_changes": sum(r.get("rgb_changed", False) for r in steps),
            "unverified_local_stops_rejected": sum(r.get("local_goal_stop_rejected", False) for r in steps),
            "global_recovery_steps": sum(r.get("environment_step_returned", False) and r.get("recovered_global_instruction", False) for r in steps),
            "autonomous_node_discovery": False,
            "autonomous_branch_choice": any(r.get("routing_status") == "task_goal_committed" for r in rounds),
            "task_goals_committed": sum(r.get("routing_status") == "task_goal_committed" for r in rounds),
            "planning_errors": sum("error" in r for r in rounds),
            "goal_policy": "current-task VLM over current visual branches and actual queried evidence" if predictive else "current episode instruction; memory is evidence only",
            "task_planner_available": predictive is not None,
            "query_mode": args.query_mode,
            "world_model_branch_queries_in_live": sum(len(r.get("queries", [])) for r in rounds),
            "supported_query_count": sum(q["supported_by_history"] for r in rounds for q in r.get("queries", [])),
            "active_observation_turns": observation_turns,
            "revisit_threshold_source": "internal fit-validation scenes",
            "arrival_or_navigation_success_not_asserted": True, "new_event_outcomes_not_written_without_arrival_verification": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=REPO / "checkpoints/LightNav-0")
    parser.add_argument("--backend", choices=("vllm_local", "hf"), default="vllm_local")
    parser.add_argument("--server", default=None)
    parser.add_argument("--live-steps", type=int, default=55)
    parser.add_argument("--planner-model", type=Path, default=None)
    parser.add_argument("--planner-device", default="cuda:0")
    parser.add_argument("--planning-interval", type=int, default=8)
    parser.add_argument("--query-mode", choices=("predicted", "direct", "shuffled"), default="predicted")
    parser.add_argument("--maximum-observation-turns", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.live_steps <= 60:
        parser.error("live steps must be in 1..60")
    if args.planning_interval < 1 or not 0 <= args.maximum_observation_turns <= 4:
        parser.error("planning interval must be positive; observation turns must be in 0..4")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = args.training_run.resolve()
    config = json.loads((root / "config.json").read_text())
    model_path, selected = select_checkpoint(root)
    save(output / "selection.json", {"model": str(model_path), "model_sha256": digest(model_path), "selection": selected,
                                      "criterion": "internal score only; same-history non-regression required"})
    predictor = load_residual_predictor(model_path)
    bank, pair_index, query, arrays = seed_prior_memory(config, predictor, output)
    base = Path(config["base_run"])
    features = raw_features(base)
    raw = np.stack([features[i] if i else np.zeros(1152) for i in query["history_image_ids"]]).astype(np.float32)
    columns = None
    for path in sorted(root.glob("columns_*.npz")):
        a = np.load(path)
        match = np.flatnonzero(a["ids"] == query["history_image_ids"][-1])
        if len(match):
            columns = torch.from_numpy(a["columns"][match])
    if columns is None:
        raise ValueError("current image has no audited spatial features")
    evidence = predict_branch_with_memory(predictor, bank, context=query["scene"], episode_id=query["episode_uid"],
                    visible_sequence=bank.latest_sequence(query["scene"]), raw_history=torch.from_numpy(raw)[None],
                    valid=torch.tensor([query["history_valid"]]), branch=torch.tensor([query["branch_direction"]]), columns=columns)
    local_index = np.flatnonzero(np.flatnonzero(~arrays["fit"]) == pair_index)[0]
    reference = np.load(model_path.parent / "dev_predictions.npy")[local_index]
    difference = float(np.max(np.abs(reference - evidence.future_feature)))
    if difference > 1e-4:
        raise ValueError(f"runtime world-model prediction differs from saved evaluation: {difference}")
    np.save(output / "predicted_future_feature.npy", evidence.future_feature)
    save(output / "world_model_query.json", {"sample": query["sample_id"], "feature_shape": [1152],
                     "used_prior_sequences": list(evidence.past_transition_sequences), "visible_sequence": evidence.visible_sequence,
                     "retrieved_observations": [{"episode": m.episode_id, "before": m.source_observation, "after": m.target_observation,
                                                  "score": m.score} for m in evidence.retrieved_evidence],
                     "runtime_vs_saved_max_difference": difference, "query_future_is_not_an_input": True,
                     "branch_direction_source": "offline teacher, explicitly limited to interface smoke"})
    print("admitting complete chronological RGB prefix for the base model", flush=True)
    admitted = source_admission(base, query["source_run"])
    episode = next(e for e in admitted["episodes"] if e["episode_uid"] == query["episode_uid"])
    observations = sorted([o for o in admitted["observations"] if o["episode_uid"] == query["episode_uid"] and
                           o["step_index"] <= query["origin_index"]], key=lambda o: o["step_index"])
    if [o["step_index"] for o in observations] != list(range(query["origin_index"] + 1)):
        raise ValueError("base-model prefix has missing observations")
    frames = [read_admitted_rgb(o["rgb"]) for o in observations]
    save(output / "prefix_admission.json", {"mode": admitted["admission_mode"], "run_dirs": [str(p) for p in admitted["run_dirs"]],
                                            "frames": len(frames), "observation_ids": [o["observation_id"] for o in observations]})
    np.save(output / "chronological_rgb_prefix.npy", np.stack(frames), allow_pickle=False)
    del admitted, observations
    engine, bundle, factory = build_base(args)
    info = {"habitat_time_step": 0.1, "lin_vel_range": [0., 2.5], "ang_vel_range": [-300., 300.]}
    outcomes = {}
    for mode in ("original_instruction", "harness_local_goal"):
        executor = HarnessLightNavExecutor(factory(info))
        executor.reset({"instruction": episode["instruction_text"]})
        for frame in frames[:-1]:
            executor.policy.agent.observe(frame)
        before = len(executor.policy.agent._history_frame_ids)
        angle = math.atan2(*query["branch_direction"])
        if mode == "harness_local_goal":
            executor.commit_goal(LocalGoal(local_instruction(angle), query["history_observation_ids"][-1],
                                 evidence.past_transition_sequences, angle), current_observation=query["history_observation_ids"][-1],
                                 visible_sequence=evidence.visible_sequence)
        started = time.monotonic()
        action, details = executor.propose_action({"rgb": frames[-1], "instruction": episode["instruction_text"]})
        after = len(executor.policy.agent._history_frame_ids)
        if after != before + 1:
            raise ValueError("goal change reset or duplicated native episode history")
        trajectory = details.get("predicted_traj")
        outcomes[mode] = serial({"tool_status": executor.last_tool_status, "action": action,
                    "goal_source": "explicit_teacher_interface_probe" if mode == "harness_local_goal" else "original_episode_task",
                    "input_instruction": executor.policy.agent.instruction, "original_instruction": executor.original_instruction,
                    "history_before_current": before, "history_after_current": after,
                    "waypoints": trajectory, "rvq_codes": details.get("rvq_codes"), "raw_text": details.get("raw_text"),
                    "latency_s": time.monotonic() - started, "valid_waypoints": trajectory is not None and np.isfinite(trajectory).all(),
                    "native_slowfast": bool(bundle.slowfast_tiers)})
        print(json.dumps({"base_mode": mode, "valid_waypoints": bool(outcomes[mode]["valid_waypoints"]), "history": after}), flush=True)
    live = None
    if args.server:
        calibration = calibrate_revisit_floor(root)
        save(output / "revisit_calibration.json", calibration)
        live = live_smoke(args, config, predictor, bank, query, factory, output, calibration)
    bank.close()
    save(output / "result.json", {"base_model_loaded": str(args.model_path.resolve()), "backend": args.backend,
              "world_model": str(model_path), "recorded_prefix_frames": len(frames), "query": query["sample_id"],
              "base_inference": outcomes, "live": live, "autonomous_harness_implemented": False,
              "environment_memory_automatically_selects_route": False,
              "semantic_task_planner_implemented": bool(live and live["task_planner_available"]),
              "prediction_is_not_a_real_memory_fact": True, "navigation_gain_not_established": True})
    print(json.dumps({"state": "complete", "output": str(output), "live": live}), flush=True)


if __name__ == "__main__":
    main()
