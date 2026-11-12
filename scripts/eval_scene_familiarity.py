"""Eight-GPU frozen-model experience-budget replay and paired live navigation.

Experience comes from admitted, strictly older real recorded episodes. New live
attempts are saved but not relabeled as verified next-event arrivals. This is a
causal experience intervention, not online parameter learning or a blind test.
An optional completed baseline fixes the same tasks for an all-history follow-up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.event_residual_data import direct_memory_prediction, materialize_memory  # noqa: E402
from experiments.world_model.event_runtime import HarnessLightNavExecutor, load_residual_predictor  # noqa: E402
from experiments.world_model.event_train import digest, metrics, read_rows, save  # noqa: E402
from experiments.world_model.familiarity import (  # noqa: E402
    budget_memory_indices,
    choose_live_scenes,
    seed_budget,
)
from experiments.world_model.online_input import online_info, online_observation  # noqa: E402
from experiments.world_model.scene_memory import PersistentSceneMemory  # noqa: E402
from scripts.smoke_event_lightnav import build_base, raw_features, select_checkpoint, serial  # noqa: E402

BUDGETS = (0, 1, 2, 4, None)
CONDITIONS = (("cold_predicted", 0, "predicted"), ("one_predicted", 1, "predicted"),
              ("many_predicted", 3, "predicted"), ("many_direct", 3, "direct"))


def load_data(config):
    root = Path(config["training_run"])
    base = Path(config["base_run"])
    pairs, arrays = read_rows(base / "pairs.jsonl"), np.load(base / "tensors.npz")
    columns = {}
    for path in ([] if config.get("baseline_run") else sorted(root.glob("columns_*.npz"))):
        with np.load(path) as chunk:
            columns.update(zip(chunk["ids"].tolist(), chunk["columns"], strict=True))
    return pairs, arrays, columns


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = args.training_run.resolve()
    training = json.loads((root / "config.json").read_text())
    base = Path(training["base_run"])
    pairs = read_rows(base / "pairs.jsonl")
    order = json.loads((root / "episode_order.json").read_text())
    scenes = choose_live_scenes(pairs, order)
    conditions = CONDITIONS
    if args.baseline_run is not None:
        previous = json.loads((args.baseline_run / "config.json").read_text())
        completed = json.loads((args.baseline_run / "results.json").read_text())
        if not completed["input_and_weights_unchanged"] or previous["training_run"] != str(root):
            raise ValueError("follow-up needs a completed unchanged baseline with the same world model")
        if previous["live_steps"] != args.live_steps:
            raise ValueError("follow-up must preserve the baseline navigation budget")
        scenes = previous["live_scenes"]
        conditions = (("all_predicted", None, "predicted"), ("all_direct", None, "direct"))
    checkpoint, _ = select_checkpoint(root)
    config = {"output": str(output), "training_run": str(root), "base_run": str(base),
              "checkpoint": str(checkpoint), "episode_order": order, "live_scenes": scenes,
              "offline_scenes": sorted({p["scene"] for p in pairs}), "workers": 8,
              "model_path": str(REPO / "checkpoints/LightNav-0"),
              "planner_model": str(REPO / "checkpoints/world_model_planner/Qwen3-VL-4B-Instruct"),
              "encoder_snapshot": training["encoder_snapshot"],
              "habitat_python": "/home/szgx/miniconda3/envs/habitat/bin/python",
              "sim_config": str(REPO / "experiments/world_model/configs/collect_train_go2w_radius010.yaml"),
              "base_port": 5760, "live_steps": args.live_steps, "planning_interval": 12,
              "maximum_observation_turns": 2,
              "offline_budgets": ["all" if b is None else b for b in BUDGETS],
              "live_conditions": [list(x) for x in conditions],
              "baseline_run": str(args.baseline_run.resolve()) if args.baseline_run is not None else None,
              "offline_artifacts_reused": args.baseline_run is not None,
              "scene_selection": "lexical dev-first scenes with >=3 recorded episodes, latest episode held fixed",
              "history_selection": "nested earliest strictly older episodes of the same scene",
              "weights_frozen": True, "new_live_attempts_do_not_become_verified_arrivals": True,
              "fit_scene_results_not_generalization": True, "development_previously_examined_not_blind": True,
              "global_STOP": "dispatch LightNav original-task STOP to simulator; local STOP is unverified feedback",
              "current_future_and_privileged_pose_goal_not_policy_input": True}
    save(output / "config.json", config)
    paths = [Path(__file__), REPO / "experiments/world_model/familiarity.py",
             REPO / "experiments/world_model/event_runtime.py", REPO / "experiments/world_model/scene_memory.py",
             REPO / "experiments/world_model/predictive_harness.py", REPO / "experiments/world_model/predictive_queries.py",
             REPO / "experiments/world_model/visual_harness.py", REPO / "experiments/world_model/task_planning.py",
             REPO / "experiments/world_model/event_anchored.py", REPO / "experiments/world_model/event_residual_data.py",
             REPO / "scripts/smoke_event_lightnav.py", Path(config["sim_config"]), checkpoint,
             root / "config.json", root / "episode_order.json", base / "config.json", base / "pairs.jsonl",
             base / "tensors.npz", base / "feature_contract.json",
             Path(config["planner_model"]) / "asset_manifest.json"]
    paths += sorted(root.glob("columns_*.npz")) + sorted(base.glob("features_*.npz"))
    save(output / "input_manifest.json", {str(p): digest(p) for p in paths})
    for path in paths:
        if path.suffix in (".py", ".yaml"):
            target = output / "source_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    print(json.dumps({"config": str(output / "config.json"), "live_scenes": scenes,
                      "live_runs": 8*len(conditions), "maximum_live_steps": 8*len(conditions)*args.live_steps}), flush=True)


@torch.inference_mode()
def replay(config, worker, output, predictor, pairs, arrays, columns):
    ids = np.array([i for i, p in enumerate(pairs) if p["scene"] in config["offline_scenes"][worker::8]])
    features = raw_features(Path(config["base_run"]))
    raw = np.stack([[features[key] if key else np.zeros(1152, dtype=np.float32)
                     for key in pairs[i]["history_image_ids"]] for i in ids]).astype(np.float32)
    spatial = np.stack([columns[pairs[i]["history_image_ids"][-1]] for i in ids]).astype(np.float32)
    predictions, direct, source_counts, compatible_counts, exposures = [], [], [], [], []
    for budget in BUDGETS:
        selected, episode_counts, sources, compatible = budget_memory_indices(
            pairs, config["episode_order"], arrays["history"], arrays["branch"], ids, budget)
        memory = materialize_memory(selected, arrays["history"], arrays["target"], arrays["branch"])
        outputs = []
        for start in range(0, len(ids), 48):
            rows = slice(start, start+48)
            inputs = [torch.from_numpy(value[rows]).cuda() for value in
                      [raw, arrays["valid"][ids], arrays["branch"][ids], spatial]]
            kwargs = {key: torch.from_numpy(value[rows]).cuda() for key, value in memory.items()}
            outputs.append(predictor(*inputs, **kwargs).cpu().numpy())
        prediction = np.concatenate(outputs)
        predictions.append(prediction)
        direct.append(direct_memory_prediction(arrays["history"][ids, -1], arrays["branch"][ids], memory))
        source_counts.append(sources)
        compatible_counts.append(compatible)
        exposures.append(episode_counts)
        print(json.dumps({"phase": "offline", "worker": worker, "budget": budget,
                          "queries": len(ids), "supported": int((compatible > 0).sum())}), flush=True)
    # The repeat has no new data and no parameter updates: it must not learn.
    selected, *_ = budget_memory_indices(pairs, config["episode_order"], arrays["history"], arrays["branch"], ids, 1)
    memory = materialize_memory(selected, arrays["history"], arrays["target"], arrays["branch"])
    count = min(8, len(ids))
    repeated = predictor(*[torch.from_numpy(value[:count]).cuda() for value in
                          [raw, arrays["valid"][ids], arrays["branch"][ids], spatial]],
                         **{k: torch.from_numpy(v[:count]).cuda() for k, v in memory.items()}).cpu().numpy()
    repeat_difference = float(np.max(np.abs(repeated-predictions[1][:count])))
    if repeat_difference > 1e-5:
        raise ValueError("identical experience changed a frozen model prediction")
    np.savez_compressed(output / "offline.npz", ids=ids, predictions=np.stack(predictions), direct=np.stack(direct),
                        source_counts=np.stack(source_counts), compatible_counts=np.stack(compatible_counts),
                        exposures=np.stack(exposures), repeat_max_difference=repeat_difference)


def live_condition(config, selected, name, budget, mode, env, predictor, factory, encoder, processor, visual, pairs, arrays, output):
    from PIL import Image

    from experiments.world_model.predictive_harness import PredictiveTaskHarness
    from experiments.world_model.visual_harness import (
        ObservationImages,
        QwenBranchProposer,
        QwenTaskPlanner,
        register_event_images,
    )
    from lightnav.velocity import first_waypoint_to_velocity_cmd

    folder = output / name
    folder.mkdir()
    (folder / "planning_images").mkdir()
    bank = PersistentSceneMemory(folder / "experience.sqlite", contract_id=predictor.feature_contract_id)
    prefix = seed_budget(bank, pairs, arrays, config["episode_order"], selected["query_index"], budget)
    # Repeated exposure to the same immutable event ids is not extra knowledge.
    seed_budget(bank, pairs, arrays, config["episode_order"], selected["query_index"], budget)
    assert bank.count(selected["scene"]) == prefix["records"]
    save(folder / "history_prefix.json", prefix)
    obs, info = env.reset()
    if str(info.get("episode_id")) != selected["episode_id"] or selected["scene"] not in info.get("scene_id", ""):
        raise ValueError("paired live reset changed scene or episode")
    initial = hashlib.sha256(np.asarray(obs["rgb"]).tobytes()).hexdigest()
    Image.fromarray(np.asarray(obs["rgb"])).save(folder / "initial.png")
    executor = HarnessLightNavExecutor(factory(online_info(info)))
    executor.reset(online_observation(obs))
    images = ObservationImages()
    register_event_images(images, Path(config["base_run"]))
    harness = PredictiveTaskHarness(executor=executor, predictor=predictor, bank=bank,
        memory_context=selected["scene"], episode_id=f"live-familiarity:{selected['episode_uid']}:{name}",
        images=images, proposer=QwenBranchProposer(visual, images), planner=QwenTaskPlanner(visual, images), query_mode=mode)
    turns, rounds, steps = 0, [], []
    call_offset = len(visual.calls)
    initial_distance = info.get("distance_to_goal")
    stop_reason = "client_budget"

    def plan(step, record):
        nonlocal turns
        images.get(harness.observations[-1]).save(folder / "planning_images" / f"step_{step:03d}.png")
        try:
            result = harness.plan()
            rounds.append({"step": step, **result.trace})
            if result.goal is not None:
                harness.report_execution(f"At step {step} selected {result.goal.instruction}; arrival not verified.")
            if result.action == "observe" and abs(result.observation_turn_rad) > 1e-6 and turns < config["maximum_observation_turns"]:
                turns += 1
                executor.restore_global_goal()
                record["active_observation"] = True
                return first_waypoint_to_velocity_cmd(np.array([0., 0., result.observation_turn_rad]),
                    executor.policy.dt, executor.policy.lin_vel_range, executor.policy.ang_vel_range)
            return None
        except (ValueError, TypeError, KeyError) as error:
            executor.restore_global_goal()
            rounds.append({"step": step, **harness.last_trace, "error": str(error), "action": "continue_global"})
            harness.report_execution(f"Tool response rejected at step {step}: {error}; original instruction resumed.")
            return None
        finally:
            save(folder / "planning.json", rounds)
            save(folder / "visual_calls.json", visual.calls[call_offset:])

    try:
        for step in range(config["live_steps"]):
            rgb = np.asarray(obs["rgb"])
            with torch.inference_mode():
                vision = encoder(processor(images=[rgb], return_tensors="pt")["pixel_values"].cuda())
            feature = vision.pooler_output.float().cpu().numpy()[0]
            columns = vision.last_hidden_state.reshape(1, 27, 27, 1152).mean(1).float().cpu()
            # Every condition has its own image registry. Keep mode/budget out
            # of the image references shown to the blinded task planner.
            reference = f"current-episode:view:{step}"
            harness.observe(reference, rgb, feature, columns)
            record = {"step": step}
            action = plan(step, record) if step % config["planning_interval"] == 0 else None
            if action is None:
                action, details = executor.propose_action(online_observation(obs), online_info(info))
                if action is None and executor.active_goal is not None and executor.last_tool_status == "stop_proposed":
                    record["local_stop_rejected"] = True
                    before = len(executor.policy.agent._history_frame_ids)
                    executor.restore_global_goal()
                    harness.report_execution(f"At step {step} local STOP was unverified; original task restored.")
                    action, details = executor.replan_observed_frame()
                    assert before == len(executor.policy.agent._history_frame_ids)
                if action is None and executor.last_tool_status == "stop_proposed":
                    action = executor.policy.stop_action()
                    record["global_stop_dispatched"] = True
                elif action is None:
                    record["tool_failure"] = executor.last_tool_status
                    stop_reason = "tool_failure"
                    steps.append(record)
                    break
                record["rvq_codes"] = details.get("rvq_codes")
            else:
                executor.policy.observe(online_observation(obs), online_info(info))
            record.update(action=serial(action), local_goal_active=executor.active_goal is not None,
                          native_history_frames=len(executor.policy.agent._history_frame_ids))
            old_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
            obs, _reward, terminated, truncated, info = env.step(action)
            record.update(environment_step_returned=True, terminated=terminated, truncated=truncated,
                          rgb_changed=old_hash != hashlib.sha256(np.asarray(obs["rgb"]).tobytes()).hexdigest())
            record["evaluation_only"] = serial({key: info[key] for key in
                ["distance_to_goal", "success", "spl", "oracle_success", "path_length"] if key in info})
            steps.append(record)
            save(folder / "steps.json", serial(steps))
            if step % 12 == 0:
                save(output / "progress.json", {"condition": name, "step": step, "history": prefix["episodes"]})
                print(json.dumps({"phase": "live", "scene": selected["scene"], "condition": name,
                                  "step": step, "queries": sum(len(r.get("queries", [])) for r in rounds)}), flush=True)
            if terminated or truncated:
                stop_reason = "global_stop" if record.get("global_stop_dispatched") else "environment_limit"
                break
        Image.fromarray(np.asarray(obs["rgb"])).save(folder / "final.png")
        evaluation = serial({key: info[key] for key in ["distance_to_goal", "success", "spl", "oracle_success", "path_length"] if key in info})
        queries = [q for r in rounds for q in r.get("queries", [])]
        result = {"condition": name, "scene": selected["scene"], "partition": selected["partition"],
                  "episode_id": selected["episode_id"], "history_episodes": prefix["episodes"], "history_records": prefix["records"],
                  "query_mode": mode, "steps": len(steps), "initial_rgb_sha256": initial,
                  "initial_distance_evaluation_only": initial_distance, "evaluation_only": evaluation,
                  "stop_reason": stop_reason, "queries": len(queries),
                  "supported_queries": sum(q["supported_by_history"] for q in queries),
                  "queries_with_actual_evidence": sum(bool(q["retrieved"]) for q in queries),
                  "task_goals": sum(r.get("routing_status") == "task_goal_committed" for r in rounds),
                  "planning_errors": sum("error" in r for r in rounds), "observation_turns": turns,
                  "unverified_local_stops_rejected": sum(r.get("local_stop_rejected", False) for r in steps),
                  "new_experience_written_from_live": False, "native_episode_history_preserved": True}
        assert bank.count(selected["scene"]) == prefix["records"]
        save(folder / "result.json", result)
        return result
    finally:
        bank.close()


def worker(config, number):
    if not torch.cuda.is_available():
        raise RuntimeError("real model experiment requires CUDA; no CPU fallback")
    torch.set_num_threads(2)
    torch.manual_seed(20261105)
    output = Path(config["output"]) / f"worker_{number:02d}"
    output.mkdir(exist_ok=False)
    pairs, arrays, columns = load_data(config)
    predictor = load_residual_predictor(Path(config["checkpoint"]), "cuda:0")
    if config.get("baseline_run"):
        shutil.copyfile(Path(config["baseline_run"]) / f"worker_{number:02d}/offline.npz", output / "offline.npz")
    else:
        replay(config, number, output, predictor, pairs, arrays, columns)
    del columns
    selected = config["live_scenes"][number]
    save(output / "episode_allowlist.json", [{"scene": selected["scene"], "episode_id": selected["episode_id"]}])
    ready = output / "habitat.ready"
    port = config["base_port"]+number
    server_env = os.environ.copy()
    server_env.pop("CUDA_VISIBLE_DEVICES", None)
    server_env.update(PYTHONPATH=str(REPO / "habitat_server"), HABITAT_SIM_GPU_ID=str(number))
    command = [config["habitat_python"], "-u", "-m", "lightnav_habitat.serve", "--task", "vlnce",
               "--config", config["sim_config"], "--split", "train", "--port", str(port),
               "--max-steps", str(config["live_steps"]), "--episode-ids", str(output / "episode_allowlist.json"),
               "--ready-file", str(ready)]
    server_log = (output / "habitat.log").open("w")
    server = subprocess.Popen(command, env=server_env, cwd=REPO, stdout=server_log, stderr=subprocess.STDOUT)
    env = None
    try:
        deadline = time.monotonic()+180
        while not ready.exists():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Habitat failed before readiness; inspect this worker's habitat.log")
            time.sleep(.5)
        from transformers import SiglipImageProcessor, SiglipVisionModel

        from experiments.world_model.feature_cache import AUDITED_PREPROCESSOR, SIGLIP_WEIGHT_SHA256
        from experiments.world_model.visual_harness import LocalQwenVision
        from lightnav.habitat.remote_env import RemoteEnvClient

        _engine, _bundle, factory = build_base(SimpleNamespace(model_path=Path(config["model_path"]), backend="vllm_local"))
        snapshot = Path(config["encoder_snapshot"])
        if digest(snapshot / "model.safetensors") != SIGLIP_WEIGHT_SHA256:
            raise ValueError("fixed SigLIP encoder changed")
        encoder = SiglipVisionModel.from_pretrained(snapshot, local_files_only=True, dtype=torch.float32).cuda().eval()
        processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
        visual = LocalQwenVision(Path(config["planner_model"]))
        env = RemoteEnvClient(f"tcp://127.0.0.1:{port}", timeout_ms=120000, recv_retries=0)
        results = []
        for name, budget, mode in config["live_conditions"]:
            row = live_condition(config, selected, name, budget, mode, env, predictor, factory, encoder,
                                 processor, visual, pairs, arrays, output)
            results.append(row)
            if row["initial_rgb_sha256"] != results[0]["initial_rgb_sha256"]:
                raise ValueError("experience-budget conditions did not reset to identical RGB")
            save(output / "live_results.json", results)
        save(output / "complete.json", {"worker": number, "runs": len(results), "state": "complete"})
    finally:
        if env is not None:
            env.close()
        if server.poll() is None:
            server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server_log.close()


def summarize(config):
    output = Path(config["output"])
    pairs = read_rows(Path(config["base_run"]) / "pairs.jsonl")
    arrays = np.load(Path(config["base_run"]) / "tensors.npz")
    shards = [np.load(output / f"worker_{i:02d}/offline.npz") for i in range(8)]
    ids = np.concatenate([r["ids"] for r in shards])
    sort = np.argsort(ids)
    assert np.array_equal(ids[sort], np.arange(len(pairs)))
    offline = []
    for kind in ["train_dev", "train_fit"]:
        subset = np.array([p["partition"] == kind for p in pairs])
        gallery = arrays["target"][subset]
        labels = arrays["target_labels"][subset]
        scenes = [p["scene"] for p in pairs if p["partition"] == kind]
        for index, budget in enumerate(config["offline_budgets"]):
            compatible = np.concatenate([r["compatible_counts"][index] for r in shards])[sort][subset]
            sources = np.concatenate([r["source_counts"][index] for r in shards])[sort][subset]
            exposures = np.concatenate([r["exposures"][index] for r in shards])[sort][subset]
            row = {"partition": kind, "budget": budget, "queries": int(subset.sum()),
                   "source_matches": int((sources > 0).sum()), "supported": int((compatible > 0).sum()),
                   "source_matched_but_direction_rejected": int(((sources > 0) & (compatible == 0)).sum()),
                   "mean_prior_episodes": float(exposures.mean())}
            for method, key in [("predicted", "predictions"), ("direct", "direct")]:
                values = np.concatenate([r[key][index] for r in shards])[sort][subset]
                row[method] = metrics(values, gallery, labels, scenes)
            offline.append(row)
    live = [row for i in range(8) for row in json.loads((output / f"worker_{i:02d}/live_results.json").read_text())]
    aggregate = []
    for partition in ["train_dev", "train_fit", "combined_exploratory"]:
        for name, _budget, _mode in config["live_conditions"]:
            rows = [r for r in live if r["condition"] == name and (partition == "combined_exploratory" or r["partition"] == partition)]
            aggregate.append({"partition": partition, "condition": name, "runs": len(rows),
                "success_rate": float(np.mean([r["evaluation_only"]["success"] for r in rows])),
                "mean_final_distance": float(np.mean([r["evaluation_only"]["distance_to_goal"] for r in rows])),
                "mean_spl": float(np.mean([r["evaluation_only"]["spl"] for r in rows])),
                "queries": sum(r["queries"] for r in rows), "supported_queries": sum(r["supported_queries"] for r in rows),
                "planning_errors": sum(r["planning_errors"] for r in rows)})
    manifest = json.loads((output / "input_manifest.json").read_text())
    unchanged = all(digest(Path(path)) == value for path, value in manifest.items())
    if not unchanged:
        raise ValueError("an input or source changed during the experiment")
    save(output / "results.json", {"offline": offline, "live": live, "live_aggregate": aggregate,
        "input_and_weights_unchanged": unchanged, "neural_parameters_updated": False,
        "offline_artifacts_reused": bool(config.get("baseline_run")),
        "duplicate_experience_repeat_max_difference": max(float(s["repeat_max_difference"]) for s in shards),
        "not_a_blind_test": True, "history_from_admitted_recordings_not_automatic_live_arrivals": True})
    print(json.dumps({"state": "complete", "output": str(output), "live_aggregate": aggregate}), flush=True)


def launch(config, config_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError("this registered protocol requires all eight CUDA GPUs")
    output = Path(config["output"])
    jobs, logs = [], []
    try:
        for number in range(8):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(number), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                       VLLM_WORKER_MULTIPROC_METHOD="spawn")
            log = (output / f"worker_{number:02d}.log").open("w")
            logs.append(log)
            jobs.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                 "--phase", "worker", "--config", str(config_path), "--worker", str(number)],
                 env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        last = None
        while any(job.poll() is None for job in jobs):
            state = [job.poll() for job in jobs]
            if any(code is not None and code != 0 for code in state):
                raise RuntimeError(f"worker failure: {state}; inspect worker logs; no results discarded")
            if state != last:
                print(json.dumps({"worker_exit_codes": state}), flush=True)
                last = state
            time.sleep(5)
        if any(job.returncode for job in jobs):
            raise RuntimeError("at least one worker failed")
        summarize(config)
    finally:
        for job in jobs:
            if job.poll() is None:
                os.killpg(job.pid, signal.SIGTERM)
        for job in jobs:
            try:
                job.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(job.pid, signal.SIGKILL)
                job.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "launch", "worker", "summarize"), required=True)
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--live-steps", type=int, default=120)
    parser.add_argument("--baseline-run", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.training_run is None or args.output is None or not 40 <= args.live_steps <= 200:
            parser.error("prepare needs training-run, output and live-steps in 40..200")
        prepare(args)
    else:
        if args.config is None or not 0 <= args.worker < 8:
            parser.error("config is required; worker must be in 0..7")
        config = json.loads(args.config.read_text())
        if args.phase == "worker":
            worker(config, args.worker)
        elif args.phase == "launch":
            launch(config, args.config.resolve())
        else:
            summarize(config)


if __name__ == "__main__":
    main()
