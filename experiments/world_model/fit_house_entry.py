"""Walk junctions in train-fit houses that have not been walked yet.

Dev houses stay untouched. A walk stops at the next junction, or at 500 steps
from the episode start. That is the same episode budget evaluation uses.
There is no shorter step cap and no meter cap. This does not train.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from experiments.world_model.branch_exit import (
    ALIGN_RAD,
    EVAL_EPISODE_STEPS,
    NEW_HOUSE_COUNT,
    arrival_stop,
    branch_velocity,
    exits_to_walk,
    follow_opening,
    is_arrival,
    motion_stalled,
    new_fit_houses,
    remaining_bearing,
    restore_error,
    restore_matches,
    search_stop,
)
from experiments.world_model.branch_exit_entry import (
    CONFIG,
    DATASET,
    HABITAT,
    INVENTORY,
    REPO,
    SIDES,
    _append,
    _depth_m,
    _free_port,
    _idle_gpu,
    _opening_count,
    _pose,
    _save_frame,
    _wait_ready,
    _write_json,
)
from experiments.world_model.decision_nodes import openings_from_depth
from experiments.world_model.group_manifest import (
    assign_groups,
    build_allowlist,
    load_split_rows,
    sha256_file,
)

OUT = REPO / "output" / "world_model_data" / "fit_houses_20261103_01"
WALKED_SCENES = {
    "5LpN3gDmAk7",
    "7y3sRwLe3Va",
    "82sE5b5pLXE",
    "EDJbREhghzL",
    "VzqfbhrpDEA",
    "cV4RVeZvu5T",
    "jh4fc5c5qoQ",
    "kEZ7cmS4wCh",
}
SERVER_STEPS = EVAL_EPISODE_STEPS


def _horizontal(a: list[float], b: list[float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5)


def _serve(house_dir: Path, gpu: str, port: int, seed: int, label: str):
    from experiments.world_model.collect import CollectClient

    ready = house_dir / f"{label}.ready"
    log_path = house_dir / f"{label}.log"
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
            str(house_dir / "allowlist.json"),
            "--scene-inventory",
            str(INVENTORY),
            "--port",
            str(port),
            "--max-steps",
            str(SERVER_STEPS),
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
    try:
        _wait_ready(proc, ready, log_path)
        return proc, CollectClient(f"tcp://127.0.0.1:{port}"), log_handle
    except Exception:
        log_handle.close()
        if proc.poll() is None:
            proc.terminate()
        raise


def _close(proc, client, log_handle) -> None:
    if client is not None:
        client.close()
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    if log_handle is not None:
        log_handle.close()


def _track(directory: Path, phase: str, position: list[float], yaw: float) -> None:
    _append(directory / "path.jsonl", {"phase": phase, "position": position, "yaw": yaw})


def _search(client, house_dir: Path, seed: int):
    obs, info = client.reset(seed)
    position, yaw = _pose(info)
    origin = position
    commands: list[dict[str, Any]] = []
    forward_moves: list[float] = []
    _track(house_dir, "search", position, yaw)
    _save_frame(house_dir, "start", obs, info)
    for step in range(EVAL_EPISODE_STEPS + 1):
        travel_m = _horizontal(position, origin)
        reason = search_stop(
            step,
            travel_m,
            _opening_count(obs),
            stalled=motion_stalled(forward_moves),
        )
        if reason == "junction":
            meters = _depth_m(obs)
            if meters is None:
                raise RuntimeError("depth missing at the junction")
            openings = exits_to_walk(openings_from_depth(meters))
            _save_frame(house_dir, "node", obs, info)
            node = {
                "position": position,
                "yaw": yaw,
                "search_steps": step,
                "openings": openings,
            }
            _write_json(house_dir / "node.json", node)
            _write_json(house_dir / "search_commands.json", {"commands": commands})
            return obs, info, node
        if reason == "no_junction":
            _write_json(house_dir / "search_commands.json", {"commands": commands})
            return None
        meters = _depth_m(obs)
        if meters is None:
            raise RuntimeError("depth missing while looking for a junction")
        decision = follow_opening("center", meters)
        if decision is None:
            command = branch_velocity(0.5)
            mode = "turn"
        else:
            command, mode = decision
        previous = position
        obs, _reward, terminated, truncated, info = client.step(
            {"wm_action_id": f"search-{step}", **command}
        )
        commands.append(command)
        position, yaw = _pose(info)
        _track(house_dir, "search", position, yaw)
        moved = _horizontal(position, previous)
        if mode == "drive":
            forward_moves.append(moved)
        else:
            forward_moves.clear()
        if terminated or truncated:
            _write_json(house_dir / "search_commands.json", {"commands": commands})
            return None
    return None


def _walk_opening(client, exit_dir: Path, opening: dict[str, Any], yaw_at_node: float, node_position: list[float], obs, info, tag: str, already_used: int) -> dict[str, Any]:
    position, yaw = _pose(info)
    facing = False
    forward_moves: list[float] = []
    reason = "truncated_budget" if already_used >= EVAL_EPISODE_STEPS else None
    travel_m = 0.0
    steps = 0
    for step in range(EVAL_EPISODE_STEPS - already_used):
        bearing = remaining_bearing(opening["bearing_left_rad"], yaw, yaw_at_node)
        if abs(bearing) <= ALIGN_RAD:
            facing = True
        if not facing:
            command = branch_velocity(bearing)
            mode = "turn"
        else:
            meters = _depth_m(obs)
            if meters is None:
                raise RuntimeError("depth missing while following the opening")
            decision = follow_opening(opening["side"], meters)
            if decision is None:
                reason = "truncated_blocked"
                break
            command, mode = decision
        previous = position
        obs, _reward, terminated, truncated, info = client.step(
            {"wm_action_id": f"{tag}-{step}", **command}
        )
        position, yaw = _pose(info)
        moved = _horizontal(position, previous)
        if mode == "drive":
            forward_moves.append(moved)
        else:
            forward_moves.clear()
        travel_m = _horizontal(position, node_position)
        _save_frame(exit_dir, f"branch-{step:03d}", obs, info, {"mode": mode, "moved_m": moved})
        reason = arrival_stop(
            already_used + step + 1,
            travel_m,
            _opening_count(obs),
            stalled=motion_stalled(forward_moves),
        )
        steps = step + 1
        if terminated or truncated:
            reason = reason or "habitat_ended"
        if reason is not None:
            break
    return {
        "exit_id": int(opening["exit_id"]),
        "side": opening["side"],
        "bearing_left_rad": float(opening["bearing_left_rad"]),
        "travel_m": travel_m,
        "branch_steps": steps,
        "stop_reason": reason,
        "arrival": is_arrival(reason),
        "restored": True,
        "error": None,
    }


def _replay(client, commands: list[dict[str, Any]], seed: int):
    obs, info = client.reset(seed)
    for index, command in enumerate(commands):
        obs, _reward, terminated, truncated, info = client.step(
            {"wm_action_id": f"replay-{index}", **command}
        )
        if terminated or truncated:
            raise RuntimeError(f"replay ended at step {index} before the junction")
    return obs, info


def _outcome(scene: str, result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"{scene}：没有采成。{result['error']}"
    if result.get("stop_reason") == "no_junction":
        return f"{scene}：走了 {result.get('search_steps', 0)} 步，没有遇到路口。"
    side = SIDES.get(result.get("side"), result.get("side"))
    where = f"{scene} 的{side}"
    if result.get("restored") is False:
        return (
            f"{where}：第二次走回路口时位置差了 {result.get('restore_place_m', 0):.2f} 米，"
            "没有再往里走。"
        )
    travel = float(result.get("travel_m") or 0.0)
    if result.get("arrival"):
        return f"{where}：走进去 {travel:.2f} 米，到了下一个路口。"
    if result.get("stop_reason") == "truncated_blocked":
        return f"{where}：走进去 {travel:.2f} 米后被挡住。"
    if result.get("stop_reason") == "truncated_budget":
        return (
            f"{where}：走进去 {travel:.2f} 米，到了评测同样的 {EVAL_EPISODE_STEPS} 步，"
            "还没到下一个路口。"
        )
    return f"{where}：走进去 {travel:.2f} 米后停下，原因是 {result.get('stop_reason')}。"


def _write_report(lines: list[str]) -> None:
    text = [
        "这批是还没走过的训练房子。每个房子先找到路口，再走进最多两个口。",
        f"走到下一个路口就停。没到的话，最多走到评测同样的 {EVAL_EPISODE_STEPS} 步，或者身体过不去。",
        "没有另外加两米或八十步的限制。走到步数上限不算到达。",
        "检查用的房子没有动。已经走过的八个训练房子也没有再采。",
        "没有用来训练，也没有做导航评测。",
        "",
        *lines,
    ]
    (OUT / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _one_house(house: dict[str, Any], gpu: str, seed: int, lines: list[str]) -> None:
    from experiments.world_model.serve_collect import prepare_server_plan

    scene = house["scene"]
    episode_id = str(house["episode_id"])
    house_dir = OUT / "houses" / f"{scene}_{episode_id}"
    house_dir.mkdir(parents=False, exist_ok=False)
    (house_dir / "assets").mkdir()
    _write_json(house_dir / "allowlist.json", house["allowlist"])
    prepare_server_plan(house_dir / "allowlist.json", DATASET, INVENTORY, partition="train_fit")
    port = _free_port()
    print(f"start scene {scene} episode {episode_id} gpu {gpu} port {port}", flush=True)
    proc = client = log_handle = None
    try:
        proc, client, log_handle = _serve(house_dir, gpu, port, seed, "search")
        found = _search(client, house_dir, seed)
        if found is None:
            commands = json.loads((house_dir / "search_commands.json").read_text(encoding="utf-8"))
            result = {
                "scene": scene,
                "episode_id": episode_id,
                "stop_reason": "no_junction",
                "search_steps": len(commands["commands"]),
                "error": None,
            }
            _append(OUT / "results.jsonl", result)
            lines.append(_outcome(scene, result))
            print(lines[-1], flush=True)
            return
        obs, info, node = found
        openings = node["openings"]
        if not openings:
            result = {
                "scene": scene,
                "episode_id": episode_id,
                "stop_reason": "no_junction",
                "search_steps": int(node["search_steps"]),
                "error": None,
            }
            _append(OUT / "results.jsonl", result)
            lines.append(_outcome(scene, result))
            print(lines[-1], flush=True)
            return
        first, *rest = openings
        exit_dir = house_dir / f"exit_{first['exit_id']}"
        exit_dir.mkdir()
        (exit_dir / "assets").mkdir()
        result = _walk_opening(
            client,
            exit_dir,
            first,
            node["yaw"],
            node["position"],
            obs,
            info,
            f"exit{first['exit_id']}",
            int(node["search_steps"]),
        )
        result.update({"scene": scene, "episode_id": episode_id})
        _write_json(exit_dir / "results.json", result)
        _append(OUT / "results.jsonl", result)
        lines.append(_outcome(scene, result))
        print(lines[-1], flush=True)
        saved_rest = rest
        saved_commands = json.loads((house_dir / "search_commands.json").read_text(encoding="utf-8"))["commands"]
        saved_node = node
    finally:
        _close(proc, client, log_handle)
    _write_report(lines)
    for opening in saved_rest:
        port = _free_port()
        print(
            f"return scene {scene} episode {episode_id} exit {opening['exit_id']} port {port}",
            flush=True,
        )
        proc = client = log_handle = None
        exit_dir = house_dir / f"exit_{opening['exit_id']}"
        exit_dir.mkdir()
        (exit_dir / "assets").mkdir()
        try:
            proc, client, log_handle = _serve(house_dir, gpu, port, seed, f"exit_{opening['exit_id']}")
            obs, info = _replay(client, saved_commands, seed)
            position, yaw = _pose(info)
            place_m, yaw_rad = restore_error(position, yaw, saved_node["position"], saved_node["yaw"])
            if not restore_matches(position, yaw, saved_node["position"], saved_node["yaw"]):
                result = {
                    "scene": scene,
                    "episode_id": episode_id,
                    "exit_id": int(opening["exit_id"]),
                    "side": opening["side"],
                    "restored": False,
                    "restore_place_m": place_m,
                    "restore_yaw_rad": yaw_rad,
                    "arrival": False,
                    "error": None,
                }
            else:
                result = _walk_opening(
                    client,
                    exit_dir,
                    opening,
                    saved_node["yaw"],
                    saved_node["position"],
                    obs,
                    info,
                    f"exit{opening['exit_id']}",
                    len(saved_commands),
                )
                result.update({
                    "scene": scene,
                    "episode_id": episode_id,
                    "restore_place_m": place_m,
                    "restore_yaw_rad": yaw_rad,
                })
        except (Exception, SystemExit) as exc:
            result = {
                "scene": scene,
                "episode_id": episode_id,
                "exit_id": int(opening["exit_id"]),
                "side": opening["side"],
                "restored": False,
                "arrival": False,
                "error": str(exc),
            }
        finally:
            _close(proc, client, log_handle)
        _write_json(exit_dir / "results.json", result)
        _append(OUT / "results.jsonl", result)
        lines.append(_outcome(scene, result))
        print(lines[-1], flush=True)
        _write_report(lines)


def collect_houses(out: Path, used_scenes: set[str], limit: int) -> None:
    """Walk up to ``limit`` train-fit houses that are not in ``used_scenes``."""
    global OUT
    OUT = out
    if not HABITAT.is_file():
        raise SystemExit(f"habitat python is missing: {HABITAT}")
    from experiments.world_model.precollection_protocol import assert_collection_not_blocked

    assert_collection_not_blocked()
    manifest_config = "experiments/world_model/configs/collect_train_go2w.yaml"
    manifest = json.loads(
        (REPO / "output/world_model_data/formal_fit_20261028_04/manifest.json").read_text(encoding="utf-8")
    )
    if sha256_file(CONFIG) != manifest["config"]["sha256"]:
        raise SystemExit("collection config no longer matches the fit run")
    if manifest["config"]["path"] != manifest_config:
        raise SystemExit("fit run config path changed")
    dataset_path, digest, rows = load_split_rows(DATASET, "train")
    grouped = assign_groups(rows)
    houses = new_fit_houses(grouped, used_scenes, limit=limit)
    if not houses:
        raise SystemExit("no new train-fit houses left")
    for house in houses:
        if house["scene"] in used_scenes:
            raise SystemExit(f"refusing an already walked house {house['scene']}")
    gpu = _idle_gpu()
    seed = int(manifest["seed"]["requested"])
    OUT.mkdir(parents=False, exist_ok=False)
    (OUT / "houses").mkdir()
    prepared = []
    for house in houses:
        prepared.append({**house, "allowlist": build_allowlist(grouped, [house], dataset_json=dataset_path, dataset_sha256=digest)})
    _write_json(
        OUT / "protocol.json",
        {
            "purpose": "find a junction in each new train-fit house and walk up to two openings",
            "dev_excluded": True,
            "walked_scenes_excluded": sorted(used_scenes),
            "houses": [
                {"scene": house["scene"], "episode_id": str(house["episode_id"])}
                for house in prepared
            ],
            "episode_steps": EVAL_EPISODE_STEPS,
            "episode_step_limit": "same as evaluation max_episode_steps",
            "shorter_step_cap": False,
            "distance_cap_m": None,
            "previous_probe_step_cap": 80,
            "previous_probe_distance_cap_m": 2.0,
            "training_use": "not_approved",
            "formal_feature_cache": False,
            "navigation_eval": False,
            "lightnav_used": False,
            "depth_used_to_choose_the_exit": False,
            "depth_used_to_keep_the_body_clear": True,
            "config_path": manifest_config,
            "config_sha256": sha256_file(CONFIG),
            "seed_requested": seed,
            "gpu": gpu,
        },
    )
    lines: list[str] = []
    for house in prepared:
        try:
            _one_house(house, gpu, seed, lines)
        except (Exception, SystemExit) as exc:
            result = {"scene": house["scene"], "episode_id": str(house["episode_id"]), "error": str(exc)}
            _append(OUT / "results.jsonl", result)
            lines.append(_outcome(house["scene"], result))
            print(lines[-1], flush=True)
        _write_report(lines)


def main() -> None:
    collect_houses(OUT, set(WALKED_SCENES), NEW_HOUSE_COUNT)


if __name__ == "__main__":
    main()
