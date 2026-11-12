"""Bounded LightNav/Harness engineering run with actual event memory and STOP verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.event_runtime import HarnessLightNavExecutor  # noqa: E402
from experiments.world_model.feature_cache import (  # noqa: E402
    AUDITED_PREPROCESSOR,
    SIGLIP_PREPROCESS_SHA256,
    SIGLIP_REVISION,
    SIGLIP_WEIGHT_SHA256,
)
from experiments.world_model.jsonutil import sha256_file  # noqa: E402
from experiments.world_model.live_images import PersistentObservationImages  # noqa: E402
from experiments.world_model.online_input import online_info, online_observation  # noqa: E402
from experiments.world_model.prediction_control import PredictionControl  # noqa: E402
from experiments.world_model.scene_memory import PersistentSceneMemory  # noqa: E402
from experiments.world_model.stateful_controller import StatefulEpisodeController, save_json  # noqa: E402
from experiments.world_model.stop_guard import configured_stop_thresholds  # noqa: E402
from experiments.world_model.tssm_runtime import TSSMSession, load_tssm_predictor  # noqa: E402
from scripts.smoke_event_lightnav import build_base, serial  # noqa: E402


def load_tasks(path):
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("tasks must be a nonempty list of scene/episode_id pairs")
    for row in rows:
        if (set(row) != {"scene", "episode_id"} or not row["scene"] or
                row["scene"] == "Vvot9Ly1tCj" or not str(row["episode_id"])):
            raise ValueError("invalid or sealed task")
    return rows


def run_base_episode(executor, env, obs, info, *, task, number, slot, max_steps, initial_rgb_hash):
    traces = []
    end_reason = "client_step_budget"
    for step in range(max_steps):
        action = serial(executor.policy.act(online_observation(obs), online_info(info)))
        obs, _, terminated, truncated, info = env.step(action)
        traces.append({"step": step, "action": action, "action_source": "LightNav_base",
            "environment_terminated": bool(terminated), "environment_truncated": bool(truncated),
            "evaluation_only": serial({k: info[k] for k in
                ["distance_to_goal", "success", "spl", "termination_reason"] if k in info})})
        save_json(slot / "steps.json", traces)
        if terminated or truncated:
            end_reason = info.get("termination_reason", "environment_truncated" if truncated else "environment_terminated")
            break
    return {"scene": task["scene"], "task": number, "dataset_episode_id": str(task["episode_id"]),
            "initial_rgb_sha256": initial_rgb_hash, "executed_steps": len(traces), "end_reason": end_reason,
            "control_mode": "base", "prediction_controlled_rounds": 0, "memory_controlled_rounds": 0,
            "evaluation_only": serial({k: info[k] for k in
                ["distance_to_goal", "success", "spl", "termination_reason"] if k in info})}


def run(args):
    from transformers import SiglipImageProcessor, SiglipVisionModel

    from experiments.world_model.visual_harness import LocalQwenVision
    from lightnav.habitat.remote_env import RemoteEnvClient

    if not torch.cuda.is_available():
        raise RuntimeError("real LightNav run requires CUDA; no fallback model or backend")
    torch.set_num_threads(2)
    tasks = load_tasks(args.tasks)
    stop_thresholds = configured_stop_thresholds(args.sim_config)
    if not args.habitat_python.is_file():
        raise ValueError("existing separate Habitat environment is required")
    if (sha256_file(args.encoder / "model.safetensors") != SIGLIP_WEIGHT_SHA256 or
            sha256_file(AUDITED_PREPROCESSOR) != SIGLIP_PREPROCESS_SHA256):
        raise ValueError("audited visual encoder or preprocessing changed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    experience = args.experience.resolve() if args.experience else output / "experience"
    experience.mkdir(parents=True, exist_ok=True)
    identifier = args.run_id or uuid.uuid4().hex[:12]
    configuration = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update(run_id=identifier, experience=str(experience),
        settings="engineering integration, not a navigation benchmark", raw_future_or_pose_inputs=False,
        official_node_or_neighbor_list_used=False, sealed_opened=False,
        measured_motion_required_for_task_progress=True,
        physical_velocity_stop_thresholds=stop_thresholds,
        scene_context="experiment namespace only; not a neural node/position input")
    save_json(output / "config.json", configuration)
    sources = [Path(__file__).resolve(), *[REPO / f"experiments/world_model/{name}.py" for name in
        ["memory_routing", "tssm_runtime", "tssm_queries", "stateful_controller", "live_images", "event_verification",
         "task_progress", "goal_verification", "task_completion", "skill_contracts", "execution_feedback", "recovery",
         "exit_crossing", "stop_guard", "prediction_control", "route_memory",
         "scene_memory", "event_tssm", "predictive_harness", "task_planning", "event_runtime",
         "entrance_alignment", "visual_harness", "online_input"]]]
    for source in sources:
        target = output / "source_snapshot" / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    save_json(output / "input_manifest.json", {str(p): sha256_file(p) for p in
        [*sources, args.world_model, args.tasks, args.sim_config, args.encoder / "model.safetensors", AUDITED_PREPROCESSOR]})
    predictor = load_tssm_predictor(args.world_model).eval()
    global_before = {k: v.clone() for k, v in predictor.state_dict().items()}
    bank = PersistentSceneMemory(experience / "facts.sqlite", contract_id=predictor.feature_contract_id,
                                 feature_dim=predictor.model.config.feature_dim)
    images = PersistentObservationImages(experience / "images")
    session = TSSMSession(predictor, bank, experience, base_hash=sha256_file(args.world_model),
        online_learning=args.online_learning, update_steps=args.update_steps, use_memory=not args.disable_memory,
        current_episode_memory=args.current_episode_memory)
    # Preserve caller-supplied order, including explicitly repeated task ids.
    # One server per task also makes readiness and teardown unambiguous.
    _engine, _bundle, factory = build_base(SimpleNamespace(model_path=args.lightnav_model, backend="vllm_local"))
    # Match model-loading/backend state across conditions, including the base
    # control. These tools are loaded but never called in base-policy mode.
    encoder = SiglipVisionModel.from_pretrained(args.encoder, local_files_only=True, dtype=torch.float32).cuda().eval()
    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    vision = LocalQwenVision(args.planner_model)
    results = []
    try:
        for number, task in enumerate(tasks):
            slot = output / f"task_{number:02d}"
            slot.mkdir()
            save_json(slot / "episode_allowlist.json", [task])
            ready, stream = slot / "habitat.ready", (slot / "habitat.log").open("w")
            environment = os.environ.copy()
            environment.pop("CUDA_VISIBLE_DEVICES", None)
            environment.update(PYTHONPATH=str(REPO / "habitat_server"), HABITAT_SIM_GPU_ID=str(args.sim_gpu))
            server = subprocess.Popen([str(args.habitat_python), "-u", "-m", "lightnav_habitat.serve", "--task", "vlnce",
                "--config", str(args.sim_config), "--split", args.dataset_split, "--port", str(args.port),
                "--max-steps", str(args.max_steps), "--episode-ids", str(slot / "episode_allowlist.json"),
                "--ready-file", str(ready)], env=environment, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            env, controller = None, None
            try:
                deadline = time.monotonic() + 180
                while not ready.exists():
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("simulator failed readiness; inspect retained habitat.log")
                    time.sleep(0.25)
                env = RemoteEnvClient(f"tcp://127.0.0.1:{args.port}", timeout_ms=120000, recv_retries=0)
                obs, info = env.reset()
                initial_rgb_hash = hashlib.sha256(np.asarray(obs["rgb"]).tobytes()).hexdigest()
                if str(info.get("episode_id")) != str(task["episode_id"]) or task["scene"] not in info.get("scene_id", ""):
                    raise ValueError("served task differs from the registered engineering task")
                executor = HarnessLightNavExecutor(factory(online_info(info)))
                executor.reset(online_observation(obs))
                if args.base_policy_only:
                    row = run_base_episode(executor, env, obs, info, task=task, number=number, slot=slot,
                        max_steps=args.max_steps, initial_rgb_hash=initial_rgb_hash)
                    save_json(slot / "result.json", row)
                    results.append(row)
                    save_json(output / "results.json", results)
                    print(json.dumps(row), flush=True)
                    continue
                memory_before = bank.count(task["scene"])
                controller = StatefulEpisodeController(session=session, executor=executor, vision=vision, images=images,
                    context=task["scene"], episode_id=f"{identifier}:episode:{number}", output=slot / "controller",
                    initial_rgb=np.asarray(obs["rgb"]), planning_interval=args.planning_interval,
                    verification_interval=args.verification_interval, observation_budget=args.observation_budget,
                    unknown_stop_policy=args.unknown_stop_policy, maximum_recovery_cycles=args.recovery_cycles,
                    depth_scale=args.depth_scale, camera_hfov=args.camera_hfov,
                    exploration_overrides=args.exploration_overrides, option_max_steps=args.option_max_steps,
                    prediction_control=PredictionControl(args.minimum_prediction_angle, args.minimum_retrieval_margin,
                                                         args.minimum_retrieval_similarity)
                                       if args.prediction_overrides else None,
                    route_memory=args.route_memory,
                    policy_stop_roundtrip=args.policy_stop_roundtrip,
                    require_measured_progress=True, stop_thresholds=stop_thresholds)
                traces, end_reason = [], "client_step_budget"
                calls_start = len(vision.calls)
                for step in range(args.max_steps + 1):
                    rgb = np.asarray(obs["rgb"])
                    with torch.inference_mode():
                        view = encoder(processor(images=[rgb], return_tensors="pt")["pixel_values"].cuda())
                    depth = np.asarray(obs["depth"]) if "depth" in obs else None
                    if depth is not None:
                        depth = np.where(depth >= 0.999, np.nan, depth)
                    controller.observe(step, rgb, view.pooler_output.float().cpu().numpy()[0],
                        view.last_hidden_state.reshape(1, 27, 27, 1152).mean(1).float().cpu(), depth)
                    if step == args.max_steps:
                        break
                    # No pose, distance, success, reference path or node list
                    # enters the controller. They may be logged after execution.
                    action, trace = controller.next_action(step, online_observation(obs), online_info(info))
                    if action is None:
                        end_reason = trace.get("end_reason", "abstained")
                        traces.append(trace)
                        break
                    action = serial(action)
                    old = hashlib.sha256(rgb.tobytes()).hexdigest()
                    obs, _reward, terminated, truncated, info = env.step(action)
                    changed = old != hashlib.sha256(np.asarray(obs["rgb"]).tobytes()).hexdigest()
                    controller.executed(step, action, rgb_changed=changed,
                                        command_id=trace.get("command_id"), action_source=trace.get("action_source"))
                    trace.update(action=action, environment_step_returned=True, rgb_changed=changed,
                        environment_terminated=bool(terminated), environment_truncated=bool(truncated),
                        evaluation_only=serial({k: info[k] for k in
                                                ["distance_to_goal", "success", "spl", "termination_reason"] if k in info}))
                    traces.append(trace)
                    controller.persist()
                    save_json(slot / "steps.json", traces)
                    if step % 12 == 0:
                        print(json.dumps({"task": number, "step": step, "facts": bank.count(task["scene"]),
                                          "action_source": trace.get("action_source", trace.get("stop_source"))}), flush=True)
                    if terminated or truncated:
                        if trace.get("stop_dispatched"):
                            end_reason = "dispatched_stop"
                        elif info.get("termination_reason") == "agent_stop":
                            end_reason = "unintended_environment_stop"
                        else:
                            end_reason = "environment_truncated" if truncated else "environment_terminated"
                        break
                # Observe the actual final returned frame even at an environment
                # limit. This can diagnose completion but cannot retroactively
                # submit STOP or turn a truncated episode into benchmark success.
                final_step = len(controller.commands)
                if controller.current_step != final_step:
                    with torch.inference_mode():
                        final_view = encoder(processor(images=[np.asarray(obs["rgb"])], return_tensors="pt")["pixel_values"].cuda())
                    final_depth = np.asarray(obs["depth"]) if "depth" in obs else None
                    if final_depth is not None:
                        final_depth = np.where(final_depth >= .999, np.nan, final_depth)
                    controller.observe(final_step, np.asarray(obs["rgb"]), final_view.pooler_output.float().cpu().numpy()[0],
                        final_view.last_hidden_state.reshape(1, 27, 27, 1152).mean(1).float().cpu(), final_depth)
                controller.check_event(final_step)
                controller.check_progress(final_step)
                controller.finish(end_reason)
                queries = [q for item in controller.planning_trace for q in item.get("queries", [])]
                row = {"scene": task["scene"], "task": number, "dataset_episode_id": str(task["episode_id"]),
                    "initial_rgb_sha256": initial_rgb_hash,
                    "executed_steps": len(controller.commands), "end_reason": end_reason,
                    "memory_before": memory_before, "memory_after": bank.count(task["scene"]),
                    "memory_direct_queries": sum(q["query_kind"] == "tssm_observed_memory_candidate" for q in queries),
                    "world_model_queries": sum(q.get("world_model_called", False) for q in queries),
                    "prediction_changed_retrieval_queries": sum(q.get("prediction_changes_top_retrieval", False) for q in queries),
                    "retrieved_outcome_queries": sum(bool(q.get("retrieved_sequences")) for q in queries),
                    "evidence_used_rounds": sum(bool(p.get("used_evidence_sequences")) for p in controller.planning_trace),
                    "prediction_controlled_rounds": sum(p.get("control_source") == "prediction" for p in controller.planning_trace),
                    "memory_controlled_rounds": sum(p.get("control_source") == "observed_memory" for p in controller.planning_trace),
                    "ambiguous_queries": sum(q.get("observation_required", False) for q in queries),
                    "verified_events": sum(r.get("event_supported", False) for r in controller.event_trace),
                    "observed_route_spans": len(controller.route_journal.records) if controller.route_journal is not None else 0,
                    "online_parameter_updates": sum(bool(r.get("writeback", {}).get("update", {}).get("parameters_changed"))
                                                    for r in controller.event_trace if r.get("writeback", {}).get("update") is not None),
                    "task_visually_supported": controller.progress.complete,
                    "policy_STOP_unverified": any(t.get("policy_STOP_deferred", False) for t in traces),
                    "semantic_arrival_confirmed_at_STOP": any(t.get("stop_dispatched", False) and t.get("semantic_arrival_confirmed", False) for t in traces),
                    "final_goal_verdict": controller.goal_verdict,
                    "final_view_diagnostic_checked": True,
                    "recovery_observation_actions": controller.recovery.total_observations,
                    "measured_motion_valid_fraction": sum(r["measurement"]["valid"] for r in controller.execution_trace)/max(1, len(controller.execution_trace)),
                    "decomposition_verbatim_valid": controller.decomposition["error"] is None,
                    "stop_interceptions": sum(t.get("local_stop_intercepted", False) or t.get("global_stop_intercepted", False) for t in traces),
                    "implicit_stop_interceptions": sum(t.get("implicit_stop_intercepted", False) for t in traces),
                    "stop_dispatched": any(t.get("stop_dispatched", False) for t in traces),
                    "evaluation_only": serial({k: info[k] for k in
                                                ["distance_to_goal", "success", "spl", "termination_reason"] if k in info}),
                    "engineering_run_not_accuracy_claim": True}
                save_json(slot / "steps.json", traces)
                save_json(slot / "visual_calls.json", vision.calls[calls_start:])
                save_json(slot / "result.json", row)
                results.append(row)
                save_json(output / "results.json", results)
                print(json.dumps(row), flush=True)
            finally:
                if controller is not None and session.episode_id is not None:
                    controller.finish("interrupted_or_error")
                if env is not None:
                    env.close()
                if server.poll() is None:
                    server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                stream.close()
        if not all(torch.equal(value, predictor.state_dict()[key]) for key, value in global_before.items()):
            raise ValueError("global world model was changed online")
        save_json(output / "complete.json", {"tasks": len(results), "global_weights_unchanged": True, "sealed_opened": False})
    finally:
        session.end_episode()
        bank.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experience", type=Path)
    parser.add_argument("--run-id", help="Optional shared reference namespace for paired conditions with independent experience banks.")
    parser.add_argument("--world-model", type=Path, default=REPO / "output/world_model_training/event_tssm_v2_20261106_01/candidate_tssm.pt")
    parser.add_argument("--lightnav-model", type=Path, default=REPO / "checkpoints/LightNav-0")
    parser.add_argument("--planner-model", type=Path, default=REPO / "checkpoints/world_model_planner/Qwen3-VL-4B-Instruct")
    parser.add_argument("--encoder", type=Path, default=Path.home() / ".cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots" / SIGLIP_REVISION)
    parser.add_argument("--habitat-python", type=Path, default=Path("/home/szgx/miniconda3/envs/habitat/bin/python"))
    parser.add_argument("--sim-config", type=Path, default=REPO / "experiments/world_model/configs/collect_train_go2w_radius010.yaml")
    parser.add_argument("--dataset-split", default="train", choices=["train", "val_seen", "val_unseen"])
    parser.add_argument("--port", type=int, default=5870)
    parser.add_argument("--sim-gpu", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500, help="Evaluation budget, 1–500 steps per task.")
    parser.add_argument("--planning-interval", type=int, default=12)
    parser.add_argument("--verification-interval", type=int, default=6)
    parser.add_argument("--observation-budget", type=int, default=8)
    parser.add_argument("--online-learning", action="store_true")
    parser.add_argument("--update-steps", type=int, default=20)
    parser.add_argument("--unknown-stop-policy", choices=["defer", "abstain"], default="defer")
    parser.add_argument("--recovery-cycles", type=int, default=3)
    parser.add_argument("--option-max-steps", type=int, default=100)
    parser.add_argument("--depth-scale", type=float, default=10.0)
    parser.add_argument("--camera-hfov", type=float, default=120.0)
    parser.add_argument("--exploration-overrides", action="store_true")
    parser.add_argument("--prediction-overrides", action="store_true", help="Allow cited, discriminative future retrieval to control a local entrance option.")
    parser.add_argument("--minimum-prediction-angle", type=float, default=10.)
    parser.add_argument("--minimum-retrieval-margin", type=float, default=.02)
    parser.add_argument("--minimum-retrieval-similarity", type=float, default=.5,
                        help="Minimum cosine similarity of a predicted future to its retrieved historical outcome.")
    parser.add_argument("--base-policy-only", action="store_true", help="Evaluate the unchanged LightNav policy with no Harness operations.")
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--current-episode-memory", action=argparse.BooleanOptionalAction, default=True,
                        help="Read already executed live facts from the current episode, still bounded by the query sequence cutoff.")
    parser.add_argument("--route-memory", action=argparse.BooleanOptionalAction, default=True,
                        help="Retain actual RGB-D route spans as retrieval evidence, separate from verified next-event arrivals.")
    parser.add_argument("--policy-stop-roundtrip", action=argparse.BooleanOptionalAction, default=True,
                        help="Latch an uncertain original-policy STOP, inspect and return in place, then dispatch that STOP.")
    args = parser.parse_args()
    if args.run_id is not None and (not args.run_id or len(args.run_id) > 128 or not all(c.isalnum() or c in "_-" for c in args.run_id)):
        parser.error("run id must be 1–128 alphanumeric, underscore or hyphen characters")
    if args.prediction_overrides:
        PredictionControl(args.minimum_prediction_angle, args.minimum_retrieval_margin,
                          args.minimum_retrieval_similarity)
    if args.base_policy_only and (args.prediction_overrides or args.exploration_overrides or args.online_learning):
        parser.error("base policy mode cannot enable Harness control or online learning")
    if (not 1 <= args.max_steps <= 500 or args.planning_interval < 1 or args.verification_interval < 1 or
            args.observation_budget < 0 or args.update_steps < 1 or not 1 <= args.port <= 65535 or args.sim_gpu < 0 or
            args.recovery_cycles < 0 or not 1 <= args.option_max_steps <= 500 or
            not 0 < args.depth_scale <= 100 or not 10 <= args.camera_hfov < 180):
        parser.error("invalid step, observation, update, port or device setting")
    run(args)
