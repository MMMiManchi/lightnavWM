"""Replay one train-fit decision node and walk the opening it did not take.

This is a collection probe. It does not train, does not write the formal
feature cache, and does not walk a dev scene. The protocol file is written
before the simulator starts.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.world_model.branch_exit import (
    ALIGN_RAD,
    BRANCH_STEPS,
    FIT_SPLIT,
    batch_passes,
    branch_stop,
    branch_velocity,
    follow_opening,
    is_arrival,
    motion_stalled,
    remaining_bearing,
    replay_actions,
    restore_error,
    restore_matches,
    unwalked_targets,
)
from experiments.world_model.decision_nodes import openings_from_depth
from experiments.world_model.ego_warp import depth_to_meters
from experiments.world_model.group_manifest import (
    assign_groups,
    build_allowlist,
    index_by_key,
    load_split_rows,
)
from experiments.world_model.jsonutil import dumps, sha256_file
from experiments.world_model.serve_collect import prepare_server_plan

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "world_model_data" / "branch_exit_20261103_03"
DONE = [("R2R_VLNCE_v1-3_preprocessed|train|VzqfbhrpDEA|151", 6, 1)]
SERVER_MAX_STEPS = 500
NODES = REPO / "output" / "world_model_diagnostics" / "decision_nodes_20261103_01" / "decision_nodes.jsonl"
FIT = REPO / "output" / "world_model_data" / "formal_fit_20261028_04"
CONFIG = REPO / "experiments" / "world_model" / "configs" / "collect_train_go2w.yaml"
INVENTORY = REPO / "experiments" / "world_model" / "splits" / "wm_scene_split_0.1.json"
DATASET = REPO / "data" / "datasets" / "R2R_VLNCE_v1-3_preprocessed"
HABITAT = Path("/home/szgx/miniconda3/envs/habitat/bin/python")
SIDES = {"left": "左边", "right": "右边", "center": "正前方"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _episode_rows(path: Path, episode_uid: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if episode_uid not in line:
                continue
            row = json.loads(line)
            if row.get("episode_uid") == episode_uid:
                rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dumps(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _idle_gpu() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        index, used = [part.strip() for part in line.split(",")]
        if index in {str(gpu) for gpu in range(2, 8)} and int(used) < 4000:
            return index
    raise RuntimeError("no idle GPU in 2..7")


def _free_port() -> int:
    for port in range(5696, 5796):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 5696..5795")


def _pose(info: dict[str, Any]) -> tuple[list[float], float]:
    pose = (info.get("wm_privileged") or {}).get("pose") or {}
    position = pose.get("world_position_xyz_m")
    yaw = pose.get("derived_yaw_rad")
    if position is None or yaw is None:
        raise RuntimeError("habitat did not return a pose")
    return [float(value) for value in position], float(yaw)


def _opening_count(obs: dict[str, Any]) -> int | None:
    depth = obs.get("depth")
    if depth is None:
        return None
    try:
        meters = depth_to_meters(np.asarray(depth))
        return len(openings_from_depth(meters))
    except (TypeError, ValueError):
        return None


def _depth_m(obs: dict[str, Any]) -> np.ndarray | None:
    depth = obs.get("depth")
    if depth is None:
        return None
    try:
        return depth_to_meters(np.asarray(depth))
    except (TypeError, ValueError):
        return None


def _save_frame(
    run_dir: Path,
    name: str,
    obs: dict[str, Any],
    info: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    rgb = np.asarray(obs["rgb"])
    depth = np.asarray(obs["depth"])
    rgb_path = run_dir / "assets" / f"{name}_rgb.npy"
    depth_path = run_dir / "assets" / f"{name}_depth.npy"
    np.save(rgb_path, rgb)
    np.save(depth_path, depth)
    position, yaw = _pose(info)
    record = {
        "name": name,
        "rgb": {"path": rgb_path.name, "sha256": sha256_file(rgb_path), "shape": list(rgb.shape)},
        "depth": {
            "path": depth_path.name,
            "sha256": sha256_file(depth_path),
            "shape": list(depth.shape),
        },
        "position": position,
        "yaw": yaw,
        "opening_count": _opening_count(obs),
    }
    if extra:
        record.update(extra)
    _append(run_dir / "observations.jsonl", record)


def _report(result: dict[str, Any]) -> str:
    target = result["target"]
    side = SIDES.get(target["side"], target["side"])
    lines = [
        "这是一次出口补采。只补训练集上已经标出的路口里、当时看见但没走的一个口。",
        "没有用来训练，也没有做导航评测。开发集没有动。",
        (
            f"这一处在场景 {target['scene']}，原记录走到第 {target['step_index']} 步。"
            f"当时没走的是{side}。"
        ),
    ]
    if result.get("error"):
        lines.append(f"采集没有走完：{result['error']}")
        return "\n".join(lines) + "\n"
    place_m = result["restore_place_m"]
    yaw_rad = result["restore_yaw_rad"]
    if not result["restored"]:
        lines.append(
            f"按原来的动作走回去，位置差了 {place_m:.2f} 米，朝向差了 {yaw_rad:.2f} 弧度。"
            "差得超过预先写下的 0.30 米或 0.30 弧度，所以没有再往那个口里走。"
        )
        return "\n".join(lines) + "\n"
    lines.append(
        f"走回去的位置差 {place_m:.2f} 米，朝向差 {yaw_rad:.2f} 弧度，在允许范围内。"
    )
    if result["arrival"]:
        lines.append(
            f"顺着这个口又走到了下一个能分岔的地方，走了 {result['travel_m']:.2f} 米，"
            f"用了 {result['branch_steps']} 步。"
        )
    elif result["stop_reason"] == "truncated_blocked":
        lines.append(
            f"顺着这个口走了 {result['travel_m']:.2f} 米后，身体前面过不去，已经停下。"
            "没有到下一个路口。"
        )
    elif result["stop_reason"] == "truncated_distance":
        lines.append(
            f"顺着这个口走进去 {result['travel_m']:.2f} 米，身体一直在往前。"
            "到了这次采集预先写下的两米，就停下了。这两米里没有再遇到下一个路口。"
        )
    else:
        lines.append(
            f"顺着这个口走了 {result['travel_m']:.2f} 米、{result['branch_steps']} 步后停下，"
            f"原因是 {result['stop_reason']}。没有到下一个路口。"
        )
    return "\n".join(lines) + "\n"


def _wait_ready(proc: subprocess.Popen, ready: Path, log_path: Path) -> None:
    deadline = time.monotonic() + 300
    while not ready.exists():
        code = proc.poll()
        if code is not None:
            tail = log_path.read_text(encoding="utf-8")[-2000:] if log_path.exists() else ""
            raise RuntimeError(f"habitat server exited {code}: {tail}")
        if time.monotonic() > deadline:
            raise RuntimeError("habitat server did not become ready")
        time.sleep(1)


def _collect(run_dir: Path, target: dict[str, Any], actions: list[dict[str, Any]], gpu: str, port: int, seed: int) -> dict[str, Any]:
    from experiments.world_model.collect import CollectClient

    ready = run_dir / "habitat.ready"
    log_path = run_dir / "habitat_server.log"
    log_handle = open(log_path, "w", encoding="utf-8")
    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = gpu
    server_env["HABITAT_SIM_GPU_ID"] = "0"
    server_env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.Popen(
        [
            str(HABITAT),
            str(REPO / "experiments" / "world_model" / "serve_collect.py"),
            "--config",
            str(CONFIG),
            "--allowlist",
            str(run_dir / "allowlist.json"),
            "--scene-inventory",
            str(INVENTORY),
            "--port",
            str(port),
            "--max-steps",
            str(target["step_index"] + BRANCH_STEPS + 5),
            "--ready-file",
            str(ready),
            "--dataset-root",
            str(DATASET),
            "--partition",
            "train_fit",
            "--seed",
            str(seed),
        ],
        cwd=str(REPO),
        env=server_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    client = None
    try:
        _wait_ready(proc, ready, log_path)
        client = CollectClient(f"tcp://127.0.0.1:{port}")
        obs, info = client.reset(seed)
        episode_id = target["episode_uid"].split("|")[-1]
        if str(info.get("episode_id")) != episode_id or target["scene"] not in str(info.get("scene_id", "")):
            raise RuntimeError(
                f"reset returned scene {info.get('scene_id')!r} episode {info.get('episode_id')!r}"
            )
        for index, action in enumerate(actions):
            obs, _reward, terminated, truncated, info = client.step(
                {"wm_action_id": f"replay-{index}", **action}
            )
            if terminated or truncated:
                raise RuntimeError(f"replay ended at step {index} before the node")
        position, yaw = _pose(info)
        place_m, yaw_rad = restore_error(position, yaw, target["position"], target["yaw"])
        restored = restore_matches(position, yaw, target["position"], target["yaw"])
        _save_frame(run_dir, "node", obs, info)
        result: dict[str, Any] = {
            "target": target,
            "gpu": gpu,
            "port": port,
            "seed": seed,
            "restore_place_m": place_m,
            "restore_yaw_rad": yaw_rad,
            "restored": restored,
            "arrival": False,
            "stop_reason": None,
            "travel_m": 0.0,
            "branch_steps": 0,
            "error": None,
        }
        if not restored:
            return result
        origin = position
        yaw_at_node = yaw
        facing = False
        forward_moves: list[float] = []
        reason = None
        for step in range(BRANCH_STEPS):
            bearing = remaining_bearing(target["bearing_left_rad"], yaw, yaw_at_node)
            if abs(bearing) <= ALIGN_RAD:
                facing = True
            if not facing:
                command = branch_velocity(bearing)
                mode = "turn"
            else:
                meters = _depth_m(obs)
                if meters is None:
                    raise RuntimeError("depth missing while following the opening")
                decision = follow_opening(target["side"], meters)
                if decision is None:
                    reason = "truncated_blocked"
                    result["stop_reason"] = reason
                    break
                command, mode = decision
            previous = position
            obs, _reward, terminated, truncated, info = client.step(
                {"wm_action_id": f"branch-{step}", **command}
            )
            position, yaw = _pose(info)
            moved_m = float(np.hypot(position[0] - previous[0], position[2] - previous[2]))
            if mode == "drive":
                forward_moves.append(moved_m)
            else:
                forward_moves.clear()
            travel_m = float(np.hypot(position[0] - origin[0], position[2] - origin[2]))
            _save_frame(
                run_dir,
                f"branch-{step:02d}",
                obs,
                info,
                {"mode": mode, "moved_m": moved_m},
            )
            opening_count = _opening_count(obs)
            reason = branch_stop(
                step + 1,
                travel_m,
                opening_count,
                stalled=motion_stalled(forward_moves),
            )
            if terminated or truncated:
                reason = reason or "habitat_ended"
            result["travel_m"] = travel_m
            result["branch_steps"] = step + 1
            result["stop_reason"] = reason
            if reason is not None:
                break
        result["arrival"] = is_arrival(reason)
        return result
    finally:
        if client is not None:
            client.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_handle.close()


def _exit_dir(target: dict[str, Any]) -> Path:
    episode_id = target["episode_uid"].split("|")[-1]
    name = f"{target['scene']}_{episode_id}_step{target['step_index']}_exit{target['exit_id']}"
    return OUT / "exits" / name


def _outcome_line(result: dict[str, Any]) -> str:
    target = result["target"]
    side = SIDES.get(target["side"], target["side"])
    where = f"{target['scene']} 第 {target['step_index']} 步的{side}"
    if result.get("error"):
        return f"{where}：没有采成。{result['error']}"
    if not result.get("restored"):
        return (
            f"{where}：走回去对不上，位置差 {result.get('restore_place_m', 0):.2f} 米，"
            "没有往里走。"
        )
    travel = float(result.get("travel_m") or 0.0)
    if result.get("arrival"):
        return f"{where}：走进去 {travel:.2f} 米，到了下一个路口。"
    if result.get("stop_reason") == "truncated_blocked":
        return f"{where}：走进去 {travel:.2f} 米后被挡住。"
    if result.get("stop_reason") == "truncated_distance":
        return f"{where}：走进去 {travel:.2f} 米，到了两米就停下。"
    if result.get("stop_reason") == "truncated_budget":
        return f"{where}：走进去 {travel:.2f} 米，步数用完了，还没到两米，也没到下一个路口。"
    return f"{where}：走进去 {travel:.2f} 米后停下，原因是 {result.get('stop_reason')}。"


def _write_report(results: list[dict[str, Any]]) -> None:
    lines = [
        "这批是训练集上其他还没走过的出口。",
        "VzqfbhrpDEA 第 6 步右边那个口已经采过，这里不再采。",
        "没有用来训练，也没有做导航评测。开发集没有动。",
        "",
    ]
    lines.extend(_outcome_line(result) for result in results)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not HABITAT.is_file():
        raise SystemExit(f"habitat python is missing: {HABITAT}")
    nodes = _read_jsonl(NODES)
    targets = unwalked_targets(nodes)
    passes = batch_passes(targets, DONE)
    if not passes:
        raise SystemExit("no remaining train-fit exit")
    manifest = json.loads((FIT / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(CONFIG) != manifest["config"]["sha256"]:
        raise SystemExit("collection config no longer matches the fit run")
    if manifest["config"]["path"] != "experiments/world_model/configs/collect_train_go2w.yaml":
        raise SystemExit("fit run config path changed")
    seed = int(manifest["seed"]["requested"])
    dataset_path, digest, rows = load_split_rows(DATASET, "train")
    grouped = assign_groups(rows)
    known = index_by_key(grouped)
    planned = [target for batch in passes for target in batch]
    for target in planned:
        if target["split"] != FIT_SPLIT:
            raise SystemExit("refusing a target outside train fit")
        scene, episode_id = target["episode_uid"].split("|")[2:]
        if known[(scene, episode_id)]["partition"] != "train_fit":
            raise SystemExit(f"{scene} {episode_id} is not train_fit")
    actions_by_key = {}
    for target in planned:
        episode_uid = target["episode_uid"]
        if episode_uid not in actions_by_key:
            actions_by_key[episode_uid] = _episode_rows(FIT / "transitions.jsonl", episode_uid)
    gpu = _idle_gpu()
    OUT.mkdir(parents=False, exist_ok=False)
    (OUT / "exits").mkdir()
    _write_json(
        OUT / "protocol.json",
        {
            "purpose": "walk the remaining openings seen at train-fit decision nodes and not taken",
            "already_collected": [
                {"episode_uid": uid, "step_index": step, "exit_id": exit_id}
                for uid, step, exit_id in DONE
            ],
            "source_run": "formal_fit_20261028_04",
            "dev_excluded": True,
            "training_use": "not_approved",
            "formal_feature_cache": False,
            "navigation_eval": False,
            "lightnav_used": False,
            "policy": "replay to the node, turn to the unwalked opening, then walk only where the body can pass",
            "truncation_is_not_an_arrival": True,
            "depth_used_to_choose_the_exit": False,
            "depth_used_to_keep_the_body_clear": True,
            "config_path": "experiments/world_model/configs/collect_train_go2w.yaml",
            "config_sha256": manifest["config"]["sha256"],
            "seed_requested": seed,
            "gpu": gpu,
            "exit_count": len(planned),
            "passes": [
                [
                    {
                        "scene": target["scene"],
                        "episode_uid": target["episode_uid"],
                        "step_index": target["step_index"],
                        "exit_id": target["exit_id"],
                        "side": target["side"],
                    }
                    for target in batch
                ]
                for batch in passes
            ],
        },
    )
    results: list[dict[str, Any]] = []
    for batch in passes:
        for target in batch:
            exit_dir = _exit_dir(target)
            exit_dir.mkdir(parents=False, exist_ok=False)
            (exit_dir / "assets").mkdir()
            scene, episode_id = target["episode_uid"].split("|")[2:]
            allowlist = build_allowlist(
                grouped,
                [known[(scene, episode_id)]],
                dataset_json=dataset_path,
                dataset_sha256=digest,
            )
            _write_json(exit_dir / "allowlist.json", allowlist)
            prepare_server_plan(exit_dir / "allowlist.json", DATASET, INVENTORY, partition="train_fit")
            actions = replay_actions(
                actions_by_key[target["episode_uid"]],
                target["episode_uid"],
                target["step_index"],
            )
            port = _free_port()
            print(
                f"start scene {target['scene']} step {target['step_index']} "
                f"exit {target['exit_id']} gpu {gpu} port {port}",
                flush=True,
            )
            try:
                result = _collect(exit_dir, target, actions, gpu, port, seed)
            except (Exception, SystemExit) as exc:
                result = {
                    "target": target,
                    "gpu": gpu,
                    "port": port,
                    "seed": seed,
                    "restored": False,
                    "arrival": False,
                    "error": str(exc),
                }
            _write_json(exit_dir / "results.json", result)
            _append(OUT / "results.jsonl", result)
            results.append(result)
            _write_report(results)
            print(_outcome_line(result), flush=True)


if __name__ == "__main__":
    main()
