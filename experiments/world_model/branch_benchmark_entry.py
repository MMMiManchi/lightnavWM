"""Collect same-state exits and the next real event after each one.

The protocol, house roles, and input hashes are written before Habitat starts.
A geometric follower is used because it can be aimed at one already-seen exit.
That policy is not LightNav. Fixed time, fixed distance, and a local stop are
not recorded as arrivals.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.world_model.branch_benchmark import (
    BRANCH_STEP_BUDGET,
    COMPLETE,
    DEVELOPMENT_HOUSES,
    FIT_HOUSES,
    LIGHTNAV_USED,
    POLICY_ID,
    SEALED_HOUSES,
    candidate_origins,
    collection_order,
    event_status,
    far_enough,
    integrity_report,
    paired_exits,
    pilot_origins,
    transition_is_complete,
)
from experiments.world_model.branch_exit import (
    ALIGN_RAD,
    FIT_SPLIT,
    branch_velocity,
    follow_opening,
    motion_stalled,
    remaining_bearing,
    replay_actions,
    restore_error,
    restore_matches,
)
from experiments.world_model.branch_supervision import build_supervision
from experiments.world_model.decision_nodes import openings_from_depth
from experiments.world_model.ego_warp import depth_to_meters
from experiments.world_model.group_manifest import (
    assign_groups,
    build_allowlist,
    index_by_key,
    load_split_rows,
)
from experiments.world_model.jsonutil import dumps, sha256_file
from experiments.world_model.latent_data import scene_partition
from experiments.world_model.serve_collect import prepare_server_plan

REPO = Path(__file__).resolve().parents[2]
NODES = REPO / "output" / "world_model_diagnostics" / "decision_nodes_20261103_01" / "decision_nodes.jsonl"
FIT = REPO / "output" / "world_model_data" / "formal_fit_20261028_04"
DEV = REPO / "output" / "world_model_data" / "formal_dev_20261028_01"
PAIRS = REPO / "output" / "world_model_training" / "event_latent_v1_20261104_01" / "pairs.jsonl"
CONFIG = REPO / "experiments" / "world_model" / "configs" / "collect_train_go2w.yaml"
INVENTORY = REPO / "experiments" / "world_model" / "splits" / "wm_scene_split_0.1.json"
DATASET = REPO / "data" / "datasets" / "R2R_VLNCE_v1-3_preprocessed"
HABITAT = Path("/home/szgx/miniconda3/envs/habitat/bin/python")
ENV_STEP_BUDGET = 30000
ORIGIN_BUDGET = 40
CONTINUATION_ORIGIN_BUDGET = 56
REPEATS = 2
SOURCE_BY_SPLIT = {
    FIT_SPLIT: FIT,
    "formal_dev_20261028_01": DEV,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dumps(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class _FileLock:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.handle: Any = None

    def __enter__(self) -> "_FileLock":
        self.output.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.output / ".write.lock", "a", encoding="utf-8")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_args: object) -> None:
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def claim_name(origin_id: str, exit_id: int, repeat_index: int) -> str:
    return f"{origin_id.replace('|', '_')}_e{int(exit_id)}_r{int(repeat_index)}"


def try_claim(output: Path, origin_id: str, exit_id: int, repeat_index: int) -> bool:
    """One execution belongs to one process. The file is the claim."""
    claims = output / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    path = claims / claim_name(origin_id, exit_id, repeat_index)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    payload = dumps(
        {
            "pid": os.getpid(),
            "origin_id": origin_id,
            "exit_id": int(exit_id),
            "repeat_index": int(repeat_index),
        }
    )
    os.write(fd, payload.encode())
    os.close(fd)
    return True


def release_stale_claims(output: Path, done: set[tuple[str, int, int]]) -> None:
    """Drop a claim whose process has exited and which has no finished result."""
    claims = output / "claims"
    if not claims.is_dir():
        return
    for path in claims.iterdir():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pid = payload.get("pid")
        key = (payload.get("origin_id"), int(payload.get("exit_id", -1)), int(payload.get("repeat_index", -1)))
        if key in done:
            continue
        if isinstance(pid, int) and _pid_alive(pid):
            continue
        path.unlink(missing_ok=True)


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


def _idle_gpu() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    free = []
    for line in completed.stdout.splitlines():
        index, used = [part.strip() for part in line.split(",")]
        if int(used) < 4000:
            free.append((int(used), int(index)))
    if not free:
        raise RuntimeError("no GPU has less than 4000 MiB in use")
    free.sort()
    return str(free[0][1])


def _free_port(gpu: str | None = None) -> int:
    if gpu is None:
        candidates = range(5796, 5896)
    else:
        base = 5900 + int(gpu) * 20
        candidates = range(base, base + 20)
    for port in candidates:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port for gpu {gpu}")


def _pose(info: dict[str, Any]) -> tuple[list[float], float]:
    pose = (info.get("wm_privileged") or {}).get("pose") or {}
    position = pose.get("world_position_xyz_m")
    yaw = pose.get("derived_yaw_rad")
    if position is None or yaw is None:
        raise RuntimeError("habitat did not return a pose")
    return [float(value) for value in position], float(yaw)


def _depth_m(obs: dict[str, Any]) -> np.ndarray | None:
    depth = obs.get("depth")
    if depth is None:
        return None
    try:
        return depth_to_meters(np.asarray(depth))
    except (TypeError, ValueError):
        return None


def _opening_count(obs: dict[str, Any]) -> int | None:
    meters = _depth_m(obs)
    if meters is None:
        return None
    try:
        return len(openings_from_depth(meters))
    except (TypeError, ValueError):
        return None


def _save_frame(run_dir: Path, name: str, obs: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    rgb = np.asarray(obs["rgb"])
    depth = np.asarray(obs["depth"])
    assets = run_dir / "assets"
    assets.mkdir(exist_ok=True)
    rgb_path = assets / f"{name}_rgb.npy"
    depth_path = assets / f"{name}_depth.npy"
    np.save(rgb_path, rgb)
    np.save(depth_path, depth)
    position, yaw = _pose(info)
    return {
        "name": name,
        "rgb_sha256": sha256_file(rgb_path),
        "depth_sha256": sha256_file(depth_path),
        "position": position,
        "yaw": yaw,
        "opening_count": _opening_count(obs),
    }


def _query_scenes(path: Path) -> list[str]:
    scenes = []
    if not path.is_file():
        return scenes
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"split": "train_dev"' not in line and '"role": "train_dev"' not in line:
                continue
            row = json.loads(line)
            if row.get("split") == "train_dev" or row.get("role") == "train_dev":
                scene = row.get("scene") or row.get("scene_id")
                if scene and scene not in scenes:
                    scenes.append(scene)
    return scenes


def exposure_record(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Prior uses of these houses. Written before this batch looks at new consequences."""
    query_scenes = _query_scenes(PAIRS)
    return {
        "decision_nodes_already_inspected": sorted({row["scene"] for row in nodes}),
        "fit_walks_already_stored": list(FIT_HOUSES),
        "development_and_sealed_walks_already_stored": list(DEVELOPMENT_HOUSES + SEALED_HOUSES),
        "closed_set_query_scenes": query_scenes,
        "sealed_house_in_closed_set_queries": any(scene in query_scenes for scene in SEALED_HOUSES),
        "sealed_claim": "已知场景的新查询封存测试",
        "this_batch_has_not_started": True,
    }


def protocol_record(origins: list[dict[str, Any]], seed: int, config_sha: str) -> dict[str, Any]:
    return {
        "goal": "G1",
        "policy_id": POLICY_ID,
        "lightnav_used": LIGHTNAV_USED,
        "event_rule": "two successive views of a junction after travelling at least 0.8 m",
        "not_arrival": ["fixed_duration", "fixed_distance", "local_stop", "official_neighbor"],
        "restore_place_m": 0.30,
        "restore_yaw_rad": 0.30,
        "branch_step_budget": BRANCH_STEP_BUDGET,
        "env_step_budget": ENV_STEP_BUDGET,
        "origin_budget": ORIGIN_BUDGET,
        "repeats_per_exit": REPEATS,
        "houses": {
            "fit": list(FIT_HOUSES),
            "development": list(DEVELOPMENT_HOUSES),
            "sealed_test": list(SEALED_HOUSES),
        },
        "pilot_origin_ids": [item["origin_id"] for item in pilot_origins(origins)],
        "origin_ids": [item["origin_id"] for item in origins],
        "seed_requested": seed,
        "config_sha256": config_sha,
        "sensor": {"radius_m": 0.10, "camera_height_m": 0.50},
        "training_use": "not_this_goal",
    }


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


def _start_server(
    run_dir: Path,
    origin: dict[str, Any],
    gpu: str,
    port: int,
    seed: int,
    allowlist: Path | None = None,
) -> tuple[subprocess.Popen, Any]:
    ready = run_dir / "habitat.ready"
    if ready.exists():
        ready.unlink()
    log_path = run_dir / "habitat_server.log"
    log_handle = open(log_path, "w", encoding="utf-8")
    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = gpu
    server_env["HABITAT_SIM_GPU_ID"] = "0"
    server_env["PYTHONDONTWRITEBYTECODE"] = "1"
    partition = "train_fit" if origin["role"] == "fit" else "train_dev"
    proc = subprocess.Popen(
        [
            str(HABITAT),
            str(REPO / "experiments" / "world_model" / "serve_collect.py"),
            "--config",
            str(CONFIG),
            "--allowlist",
            str(allowlist or (run_dir / "allowlist.json")),
            "--scene-inventory",
            str(INVENTORY),
            "--port",
            str(port),
            "--max-steps",
            str(int(origin["step_index"]) + BRANCH_STEP_BUDGET + 5),
            "--ready-file",
            str(ready),
            "--dataset-root",
            str(DATASET),
            "--partition",
            partition,
            "--seed",
            str(seed),
        ],
        cwd=str(REPO),
        env=server_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    proc._log_handle = log_handle  # type: ignore[attr-defined]
    _wait_ready(proc, ready, log_path)
    from experiments.world_model.collect import CollectClient

    return proc, CollectClient(f"tcp://127.0.0.1:{port}")


def _stop_server(proc: subprocess.Popen | None, client: Any) -> None:
    if client is not None:
        client.close()
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    handle = getattr(proc, "_log_handle", None) if proc is not None else None
    if handle is not None:
        handle.close()


def _walk_exit(client: Any, origin: dict[str, Any], exit_row: dict[str, Any], actions: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    obs, info = client.reset(int(origin["seed"]))
    episode_id = origin["episode_uid"].split("|")[-1]
    if str(info.get("episode_id")) != episode_id or origin["scene"] not in str(info.get("scene_id", "")):
        raise RuntimeError(
            f"reset returned scene {info.get('scene_id')!r} episode {info.get('episode_id')!r}"
        )
    replay_steps = 0
    for index, action in enumerate(actions):
        obs, _reward, terminated, truncated, info = client.step({"wm_action_id": f"replay-{index}", **action})
        replay_steps += 1
        if terminated or truncated:
            raise RuntimeError(f"replay ended at step {index} before the node")
    position, yaw = _pose(info)
    place_m, yaw_rad = restore_error(position, yaw, origin["position"], origin["yaw"])
    restored = restore_matches(position, yaw, origin["position"], origin["yaw"])
    node_frame = _save_frame(run_dir, "node", obs, info)
    append_jsonl(run_dir / "observations.jsonl", node_frame)
    record: dict[str, Any] = {
        "origin_id": origin["origin_id"],
        "role": origin["role"],
        "scene": origin["scene"],
        "episode_uid": origin["episode_uid"],
        "step_index": origin["step_index"],
        "exit_id": exit_row["exit_id"],
        "repeat_index": origin["repeat_index"],
        "policy_id": POLICY_ID,
        "lightnav_used": False,
        "prefix_sha256": origin["prefix_sha256"],
        "house_partition": origin.get("house_partition"),
        "route_hash_partition": origin.get("route_hash_partition"),
        "restore_place_m": place_m,
        "restore_yaw_rad": yaw_rad,
        "restored": restored,
        "pre_decision": {
            "observation_id": origin["observation_id"],
            "rgb_sha256": node_frame["rgb_sha256"],
            "position": position,
            "yaw": yaw,
            "chosen_exit": {
                "exit_id": exit_row["exit_id"],
                "bearing_left_rad": exit_row["bearing_left_rad"],
                "side": exit_row["side"],
            },
        },
        "status": "restore_failed",
        "event_complete": False,
        "travel_m": 0.0,
        "branch_steps": 0,
        "replay_steps": replay_steps,
        "env_steps": replay_steps,
        "evidence_dir": str(run_dir.relative_to(origin["output"])),
    }
    if not restored:
        return record
    counts: list[int | None] = []
    forward_moves: list[float] = []
    facing = False
    yaw_at_node = yaw
    start = position
    status = "truncated_unconfirmed"
    for step in range(BRANCH_STEP_BUDGET):
        bearing = remaining_bearing(exit_row["bearing_left_rad"], yaw, yaw_at_node)
        if abs(bearing) <= ALIGN_RAD:
            facing = True
        if not facing:
            command = branch_velocity(bearing)
            mode = "turn"
        else:
            meters = _depth_m(obs)
            if meters is None:
                status = "truncated_unconfirmed"
                break
            decision = follow_opening(exit_row["side"], meters)
            if decision is None:
                status = "truncated_blocked"
                break
            command, mode = decision
        append_jsonl(run_dir / "commands.jsonl", {"step": step, "mode": mode, "command": command})
        previous = position
        obs, _reward, terminated, truncated, info = client.step({"wm_action_id": f"branch-{step}", **command})
        position, yaw = _pose(info)
        frame = _save_frame(run_dir, f"branch-{step:03d}", obs, info)
        append_jsonl(run_dir / "observations.jsonl", frame)
        moved_m = float(np.hypot(position[0] - previous[0], position[2] - previous[2]))
        if mode == "drive":
            forward_moves.append(moved_m)
        else:
            forward_moves.clear()
        travel_m = float(np.hypot(position[0] - start[0], position[2] - start[2]))
        counts.append(frame["opening_count"])
        record["travel_m"] = travel_m
        record["branch_steps"] = step + 1
        record["env_steps"] = replay_steps + step + 1
        stalled = motion_stalled(forward_moves)
        habitat_ended = bool(terminated or truncated)
        status = event_status(
            restored=True,
            travel_m=travel_m,
            opening_counts=counts,
            stalled=stalled,
            steps=step + 1,
            habitat_ended=habitat_ended,
        )
        if status != "truncated_unconfirmed":
            break
    record["status"] = status
    record["event_complete"] = status == COMPLETE
    record["opening_counts_tail"] = counts[-2:]
    return record


def _prepare_origin(output: Path, origin: dict[str, Any], known: dict, grouped: list, dataset_path: Path, digest: str, seed: int) -> dict[str, Any]:
    source_root = SOURCE_BY_SPLIT[origin["split"]]
    actions = replay_actions(
        _episode_rows(source_root / "transitions.jsonl", origin["episode_uid"]),
        origin["episode_uid"],
        int(origin.get("source_step_index", origin["step_index"])),
    )
    parent = origin.get("parent_evidence")
    if parent:
        command_path = output / parent / "commands.jsonl"
        for line in command_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                actions.append(json.loads(line)["command"])
        if len(actions) != int(origin["step_index"]):
            raise RuntimeError(f"{origin['origin_id']} replay length is {len(actions)}")
    scene, episode_id = origin["episode_uid"].split("|")[2:]
    row = known[(scene, episode_id)]
    expected = "train_fit" if origin["role"] == "fit" else "train_dev"
    house = scene_partition(scene)
    if house != expected:
        raise RuntimeError(f"{origin['origin_id']} house partition is {house}")
    run_dir = output / "origins" / origin["origin_id"].replace("|", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    allowlist = build_allowlist(
        grouped,
        [row],
        dataset_json=dataset_path,
        dataset_sha256=digest,
    )
    write_json(run_dir / "allowlist.json", allowlist)
    prepare_server_plan(
        run_dir / "allowlist.json",
        DATASET,
        INVENTORY,
        partition=expected,
    )
    prepared = dict(origin)
    prepared.update(
        {
            "actions": actions,
            "prefix_sha256": hashlib.sha256(dumps(actions).encode()).hexdigest(),
            "run_dir": run_dir,
            "seed": seed,
            "output": output,
            "house_partition": house,
            "route_hash_partition": row["partition"],
        }
    )
    return prepared


def _outcome_line(record: dict[str, Any]) -> str:
    if record["role"] == "sealed_test":
        return "封存测试有一条执行记录。报告不写它的画面、距离和出口内容。"
    if "exit_id" not in record or "repeat_index" not in record:
        return f"{record['scene']}：这个起点没有执行。{record.get('error', '')}"
    where = f"{record['scene']} 第 {record['step_index']} 步，出口 {record['exit_id']}，第 {record['repeat_index'] + 1} 次"
    if record.get("error"):
        return f"{where}：执行失败。{record['error']}"
    if record["status"] == COMPLETE:
        return f"{where}：走出 {record['travel_m']:.2f} 米后，连续两帧仍是路口。"
    if record["status"] == "restore_failed":
        return (
            f"{where}：回到起点差了 {record.get('restore_place_m', 0):.2f} 米，"
            "没有往出口里走。"
        )
    return f"{where}：走出 {float(record.get('travel_m') or 0):.2f} 米，结果是 {record['status']}。"


def write_report(output: Path, records: list[dict[str, Any]], extra: list[str] | None = None) -> None:
    report = integrity_report([row for row in records if "origin_id" in row and "role" in row])
    write_json(output / "integrity.json", report)
    complete = [row for row in records if transition_is_complete(row)]
    by_role: dict[str, int] = {}
    origins: dict[str, set[str]] = {}
    for row in complete:
        by_role[row["role"]] = by_role.get(row["role"], 0) + 1
        origins.setdefault(row["role"], set()).add(row["origin_id"])
    lines = [
        "这批是覆盖探索：同一起点的不同出口都走，不要求沿着任务的正确路线。",
        "同一批执行分成两类监督。节点监督标的是哪里像决策点、哪里不像。后果监督标的是选定出口之后实际遇到什么。",
        "这些动作不是当前指令下的专家动作，不能拿去模仿训练导航。",
        "控制器是几何出口跟随，不是 LightNav。固定时长、固定距离和局部停止都没有当成到达。",
        "被挡住的出口只记录阻挡，不记成一条通向下一地点的连接。",
        "封存测试只写数量，不写画面和后果。",
        "",
        f"完整事件转移 {len(complete)} 条。目标是至少 80 条。",
        (
            f"fit {by_role.get('fit', 0)} 条、{len(origins.get('fit', set()))} 个起点；"
            f"development {by_role.get('development', 0)} 条、{len(origins.get('development', set()))} 个起点；"
            f"sealed test {by_role.get('sealed_test', 0)} 条、{len(origins.get('sealed_test', set()))} 个起点。"
        ),
        f"完整性检查：{'通过' if report['passed'] else '没有通过'}。",
    ]
    if report["failures"]:
        lines.extend(f"- {item}" for item in report["failures"])
    if extra:
        lines.extend(["", *extra])
    lines.extend(["", "fit 和 development 的执行记录："])
    for row in records:
        if row.get("role") == "sealed_test":
            continue
        if "status" not in row and "error" not in row:
            continue
        lines.append(_outcome_line(row))
    sealed_count = sum(1 for row in records if row.get("role") == "sealed_test")
    lines.extend(["", f"封存测试执行记录 {sealed_count} 条，其中完整事件 {by_role.get('sealed_test', 0)} 条。"])
    fit_n = by_role.get("fit", 0)
    dev_n = by_role.get("development", 0)
    sealed_n = by_role.get("sealed_test", 0)
    if len(complete) < 80 or fit_n < 48 or dev_n < 16 or sealed_n < 16:
        lines.append(
            f"合计 {len(complete)} 条。分项要求是 fit 至少 48、development 至少 16、封存测试至少 16。"
            "没有同时达到。本次 Goal 没有通过，标准没有放宽。"
        )
    else:
        lines.append("合计和分项数量都达到预先写下的要求。房屋隔离和完整性检查仍需对照本报告。")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def exception_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text or type(exc).__name__


def finished_executions(records: list[dict[str, Any]]) -> set[tuple[str, int, int]]:
    """Executions that already have a real result. A reset error can be retried."""
    done = set()
    for record in records:
        if record.get("status") in {None, "error"}:
            continue
        if "exit_id" not in record or "repeat_index" not in record:
            continue
        done.add((record["origin_id"], int(record["exit_id"]), int(record["repeat_index"])))
    return done


def execution_dir(parent: Path, repeat_index: int, exit_id: int) -> Path:
    stem = f"repeat{repeat_index}_exit{exit_id}"
    path = parent / stem
    attempt = 1
    while path.exists():
        attempt += 1
        path = parent / f"{stem}_retry{attempt}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def _frames_for(output: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    path = output / str(record.get("evidence_dir") or "") / "observations.jsonl"
    if not path.is_file():
        return []
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            frames.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return frames


def export_supervision(output: Path) -> dict[str, Any]:
    """Write node and consequence labels beside the run. Does not start Habitat."""
    records = []
    path = output / "transitions.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    nodes, consequences, counts = build_supervision(records, lambda record: _frames_for(output, record))
    dest = output / "supervision"
    dest.mkdir(exist_ok=True)
    for name, rows in (("node_frames.jsonl", nodes), ("consequences.jsonl", consequences)):
        target = dest / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")
        os.replace(temporary, target)
    counts["event_rule_unchanged"] = True
    counts["instruction_action_label"] = False
    write_json(dest / "counts.json", counts)
    write_json(
        dest / "protocol.json",
        {
            "version": "g1-supervision/20261105-02",
            "event_completion_rule": "unchanged",
            "streams": ["node_discovery", "future_consequence"],
            "instruction_action_label": False,
            "follows_task_route_required": False,
            "blocked_execution_creates_link": False,
            "official_id_used_as_class": False,
        },
    )
    return counts


def verify_output(output: Path) -> dict[str, Any]:
    records = read_jsonl(output / "transitions.jsonl")
    report = integrity_report(records)
    write_json(output / "integrity.json", report)
    return report


def disk_progress(output: Path) -> tuple[list[dict[str, Any]], set[tuple[str, int, int]], int, set[str]]:
    records = []
    path = output / "transitions.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    done = finished_executions(records)
    env_steps = sum(int(row.get("env_steps") or 0) for row in records)
    seen = {row["origin_id"] for row in records if row.get("origin_id")}
    return records, done, env_steps, seen


def commit_record(output: Path, record: dict[str, Any], notes: list[str]) -> None:
    with _FileLock(output):
        append_jsonl(output / "transitions.jsonl", record)
        if not transition_is_complete(record):
            append_jsonl(output / "failures.jsonl", record)
        records, _done, env_steps, _seen = disk_progress(output)
        write_report(output, records, notes + [f"已用环境步数 {env_steps}。"])


def _arrival_origins(output: Path) -> list[dict[str, Any]]:
    """Junctions reached by a finished exit. Sealed arrivals stay out of this pass."""
    path = output / "transitions.jsonl"
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    nodes = read_jsonl(NODES)
    occupied = [
        (row["scene"], [float(value) for value in row["position"]])
        for row in nodes
        if row["scene"] in set(FIT_HOUSES + DEVELOPMENT_HOUSES)
    ]
    kept_positions: list[list[float]] = []
    chosen = []
    ordered = sorted(
        (
            row
            for row in records
            if row.get("status") == COMPLETE
            and row.get("role") in {"fit", "development"}
            and row.get("evidence_dir")
        ),
        key=lambda row: (0 if row["role"] == "development" else 1, row["scene"], int(row["step_index"])),
    )
    for record in ordered:
        replay_steps = int(record["step_index"]) + int(record["branch_steps"])
        if replay_steps + BRANCH_STEP_BUDGET + 5 > 500:
            continue
        evidence = output / record["evidence_dir"]
        frames = _frames_for(output, record)
        if len(frames) < 2:
            continue
        last = frames[-1]
        depth_path = evidence / "assets" / f"{last['name']}_depth.npy"
        if not depth_path.is_file():
            continue
        try:
            openings = openings_from_depth(depth_to_meters(np.load(depth_path)))
        except (OSError, TypeError, ValueError):
            continue
        exits = paired_exits(openings)
        if exits is None:
            continue
        position = [float(value) for value in last["position"]]
        same_scene = [item[1] for item in occupied if item[0] == record["scene"]]
        if not far_enough(position, same_scene + kept_positions):
            continue
        commands = evidence / "commands.jsonl"
        if not commands.is_file() or sum(1 for line in commands.read_text(encoding="utf-8").splitlines() if line.strip()) != int(record["branch_steps"]):
            continue
        kept_positions.append(position)
        occupied.append((record["scene"], position))
        episode_uid = record["episode_uid"]
        chosen.append(
            {
                "origin_id": (
                    f"{record['scene']}|{episode_uid}|arrival|"
                    f"{record['step_index']}|{record['exit_id']}|{record['repeat_index']}"
                ),
                "role": record["role"],
                "scene": record["scene"],
                "episode_uid": episode_uid,
                "step_index": replay_steps,
                "source_step_index": int(record["step_index"]),
                "observation_id": f"{record.get('pre_decision', {}).get('observation_id', '')}|{last['name']}",
                "position": position,
                "yaw": float(last["yaw"]),
                "split": FIT_SPLIT if record["role"] == "fit" else "formal_dev_20261028_01",
                "exits": exits,
                "parent_evidence": record["evidence_dir"],
            }
        )
    return chosen


def write_arrival_supplement(output: Path) -> list[dict[str, Any]]:
    """Freeze the new junctions before another simulator starts."""
    origins = _arrival_origins(output)
    development = [row for row in origins if row["role"] == "development"][:6]
    fit = [row for row in origins if row["role"] == "fit"][:10]
    chosen = development + fit
    path = output / "supplement_origins.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(dumps(row) + "\n" for row in chosen), encoding="utf-8")
    os.replace(temporary, path)
    write_json(
        output / "supplement_protocol.json",
        {
            "version": "g1-arrivals/20261105-03",
            "event_completion_rule": "unchanged",
            "reason": "role quotas are short, so junctions reached by collected exits become new origins",
            "houses_unchanged": True,
            "sealed_arrivals_included": False,
            "origin_budget": CONTINUATION_ORIGIN_BUDGET,
            "origin_ids": [row["origin_id"] for row in chosen],
        },
    )
    return chosen


def _load_supplement(output: Path) -> list[dict[str, Any]]:
    path = output / "supplement_origins.jsonl"
    if not path.is_file():
        return []
    return read_jsonl(path)


def collect(output: Path, gpu: str | None = None) -> None:
    if not HABITAT.is_file():
        raise SystemExit(f"habitat python is missing: {HABITAT}")
    if not roles_disjoint_ok():
        raise SystemExit("house roles overlap")
    nodes = read_jsonl(NODES)
    node_by_id = {f"{row['scene']}|{row['episode_uid']}|{row['step_index']}": row for row in nodes}
    houses = FIT_HOUSES + DEVELOPMENT_HOUSES + SEALED_HOUSES
    origins = collection_order(candidate_origins(nodes, houses))
    known_ids = {item["origin_id"] for item in origins}
    origins.extend(item for item in _load_supplement(output) if item["origin_id"] not in known_ids)
    origin_budget = ORIGIN_BUDGET
    supplement_protocol = output / "supplement_protocol.json"
    if supplement_protocol.is_file():
        origin_budget = int(json.loads(supplement_protocol.read_text(encoding="utf-8"))["origin_budget"])
    if len(pilot_origins(origins)) < 2:
        raise SystemExit("the two fit pilot origins are not available")
    manifest = json.loads((FIT / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(CONFIG) != manifest["config"]["sha256"]:
        raise SystemExit("collection config no longer matches the fit run")
    seed = int(manifest["seed"]["requested"])
    dataset_path, digest, rows = load_split_rows(DATASET, "train")
    grouped = assign_groups(rows)
    known = index_by_key(grouped)
    resuming = (output / "protocol.json").is_file()
    if not resuming:
        if output.exists():
            raise SystemExit("output exists without a protocol")
        output.mkdir(parents=False, exist_ok=False)
        (output / "origins").mkdir()
        write_json(output / "protocol.json", protocol_record(origins, seed, manifest["config"]["sha256"]))
        write_json(output / "exposure.json", exposure_record(nodes))
        write_json(
            output / "input_hashes.json",
            {
                "decision_nodes": sha256_file(NODES),
                "config": sha256_file(CONFIG),
                "fit_manifest": sha256_file(FIT / "manifest.json"),
                "dev_manifest": sha256_file(DEV / "manifest.json"),
                "branch_benchmark": sha256_file(REPO / "experiments/world_model/branch_benchmark.py"),
                "branch_exit": sha256_file(REPO / "experiments/world_model/branch_exit.py"),
            },
        )
        write_json(
            output / "split.json",
            {
                "fit": [item["origin_id"] for item in origins if item["role"] == "fit"],
                "development": [item["origin_id"] for item in origins if item["role"] == "development"],
                "sealed_test": [item["origin_id"] for item in origins if item["role"] == "sealed_test"],
            },
        )
        write_json(
            output / "sealed_manifest.json",
            {
                "claim": "已知场景的新查询封存测试",
                "origin_count": sum(item["role"] == "sealed_test" for item in origins),
            },
        )
        records = []
    else:
        records = read_jsonl(output / "transitions.jsonl")
    with _FileLock(output):
        records, done, env_steps, seen_origins = disk_progress(output)
        release_stale_claims(output, done)
    pilot_ids = {item["origin_id"] for item in pilot_origins(origins)}
    pilot_restored = sum(1 for row in records if row.get("restored") and row.get("origin_id") in pilot_ids)
    if gpu is None:
        gpu = _idle_gpu()
    notes = [
        f"使用 GPU {gpu}。协议在模拟器启动前已写入。",
        "同一次仿真不能开始下一条路线，所以每条出口单独启动一次。",
        "多张卡各自领取还没有结果的出口，不重复执行已经完成的记录。",
    ]
    for origin in origins:
        with _FileLock(output):
            records, done, env_steps, seen_origins = disk_progress(output)
        pending = [
            (repeat_index, exit_row)
            for repeat_index in range(REPEATS)
            for exit_row in origin["exits"]
            if (origin["origin_id"], int(exit_row["exit_id"]), repeat_index) not in done
        ]
        if not pending:
            continue
        if len(seen_origins) >= origin_budget and origin["origin_id"] not in seen_origins:
            notes.append("到达本批起点预算，后面的起点没有再执行。")
            break
        if env_steps >= ENV_STEP_BUDGET:
            notes.append("到达环境步数预算，后面的出口没有再执行。")
            break
        if origin["origin_id"] not in pilot_ids and pilot_restored < 1:
            notes.append("小试没有一次成功回到起点，后面的起点没有再执行。")
            break
        source = node_by_id.get(origin["origin_id"])
        origin = dict(origin)
        if source is not None:
            origin["split"] = source["split"]
        elif "split" not in origin:
            raise KeyError(origin["origin_id"])
        with _FileLock(output):
            if not try_claim(output, origin["origin_id"], -1, -1):
                continue
        try:
            prepared = _prepare_origin(output, origin, known, grouped, dataset_path, digest, seed)
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            failed = {
                **origin,
                "error": exception_text(exc),
                "status": "error",
                "event_complete": False,
                "policy_id": POLICY_ID,
                "lightnav_used": False,
                "prefix_sha256": "",
                "pre_decision": {"observation_id": origin["observation_id"]},
            }
            commit_record(output, failed, notes)
            continue
        (output / "claims" / claim_name(origin["origin_id"], -1, -1)).unlink(missing_ok=True)
        for repeat_index, exit_row in pending:
            key = (prepared["origin_id"], int(exit_row["exit_id"]), repeat_index)
            with _FileLock(output):
                _records, done, env_steps, seen_origins = disk_progress(output)
                if env_steps >= ENV_STEP_BUDGET:
                    break
                if len(seen_origins) >= origin_budget and prepared["origin_id"] not in seen_origins:
                    break
                if key in done or not try_claim(output, *key):
                    continue
            run_dir = execution_dir(prepared["run_dir"], repeat_index, int(exit_row["exit_id"]))
            prepared["repeat_index"] = repeat_index
            port = _free_port(gpu)
            proc = None
            client = None
            try:
                proc, client = _start_server(
                    run_dir,
                    prepared,
                    gpu,
                    port,
                    seed,
                    prepared["run_dir"] / "allowlist.json",
                )
                record = _walk_exit(client, prepared, exit_row, prepared["actions"], run_dir)
            except (Exception, SystemExit) as exc:
                record = {
                    "origin_id": prepared["origin_id"],
                    "role": prepared["role"],
                    "scene": prepared["scene"],
                    "episode_uid": prepared["episode_uid"],
                    "step_index": prepared["step_index"],
                    "exit_id": exit_row["exit_id"],
                    "repeat_index": repeat_index,
                    "policy_id": POLICY_ID,
                    "lightnav_used": False,
                    "prefix_sha256": prepared["prefix_sha256"],
                    "pre_decision": {"observation_id": prepared["observation_id"]},
                    "status": "error",
                    "event_complete": False,
                    "error": exception_text(exc),
                    "env_steps": 0,
                }
            finally:
                _stop_server(proc, client)
            print(_outcome_line(record), flush=True)
            commit_record(output, record, notes)
        else:
            continue
        break
    with _FileLock(output):
        records, _done, env_steps, seen_origins = disk_progress(output)
        notes.append(f"已涉及 {len(seen_origins)} 个起点，环境步数 {env_steps}。")
        export_supervision(output)
        write_report(output, records, notes)


def roles_disjoint_ok() -> bool:
    from experiments.world_model.branch_benchmark import roles_are_disjoint

    return roles_are_disjoint()
