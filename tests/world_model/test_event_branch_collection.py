import json

import numpy as np

from experiments.world_model import event_branch_collection as collection
from experiments.world_model.branch_curation import FloorPolygon
from experiments.world_model.event_collection_control import StepAccounting


def test_exit_commit_requires_forward_progress_in_chosen_direction():
    assert collection.entered_exit([0, 0.1, -1.05], [0, 0.1, 0], 0, [0, 0.1, -1.5])
    assert not collection.entered_exit([1.05, 0.1, 0], [0, 0.1, 0], 0, [0, 0.1, -1.5])


def test_recovery_attempts_reset_after_real_progress_including_stairs():
    state = {"recovery_count": 2, "recovery_anchor": [0, 0, 0]}
    collection.reset_local_recovery(state, [0, 0, 0])
    assert state["recovery_count"] == 2
    collection.reset_local_recovery(state, [0, 0.6, 0.5])
    assert state["recovery_count"] == 0


def fake_collection(monkeypatch, tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    folder = root / "execution"
    folder.mkdir()
    StepAccounting(root).initialize()
    command = {"action": "velocity_control", "action_args": {"linear_velocity": -0.6, "angular_velocity": 0}}

    class Env:
        step_count = 0
        pause_at = None

        def step(self, action):
            self.step_count += 1
            if self.step_count == self.pause_at:
                (root / "PAUSE").touch()
            info = {"wm_privileged": {"pose": {"world_position_xyz_m": [0, 0.1, -self.step_count * 0.05],
                                                 "derived_yaw_rad": 0}},
                    "wm_execution": {"executed": True, "replayed": False}}
            return {"rgb": np.ones((2, 2, 3), dtype=np.uint8)}, 0, False, False, info

    def place(env, origin, dest, tag):
        env.step_count = 0
        node = {"position": [0, 0.1, 0], "yaw": 0, "name": "node", "rgb_path": "/fake",
                "rgb_sha256": "source"}
        history = [{**node, "name": f"history-{i}"} for i in range(7)] + [node]
        (dest / "source_history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        return {"rgb": np.ones((2, 2, 3), dtype=np.uint8)}, {
            "wm_privileged": {"pose": {"world_position_xyz_m": node["position"], "derived_yaw_rad": 0}}}, history

    def save(dest, name, obs, info):
        position, yaw = collection._pose(info)
        return {"name": name, "position": position, "yaw": yaw, "rgb_sha256": "future"}

    monkeypatch.setattr(collection, "place_and_observe", place)
    monkeypatch.setattr(collection, "_save_frame", save)
    monkeypatch.setattr(collection, "rotation_coefficients", lambda env: [0, 0, 0, 1])
    monkeypatch.setattr(collection, "reactive_command", lambda *args: (command, "follow"))
    monkeypatch.setattr(collection, "load_floor_polygons", lambda _: [FloorPolygon(
        0, "h", 0, ((-5, -100), (5, -100), (5, 100), (-5, 100)))])
    task = {"task_id": "test", "origin": {"scene": "allowed", "origin_id": "origin", "role": "fit",
                                         "expected_rgb_sha256": "source"},
            "exit": {"exit_id": 0, "bearing_left_rad": 0}, "repeat_index": 0}
    return Env(), task, folder, root


def test_1000_steps_is_censored_with_all_commands_retained(monkeypatch, tmp_path):
    env, task, folder, root = fake_collection(monkeypatch, tmp_path)
    result = collection.collect_one(env, task, folder, root, [], 1000)
    assert result["status"] == "censored_max_steps"
    assert result["branch_steps"] == 1000 and not result["event_complete"]
    assert len(collection.read_jsonl(folder / "committed.jsonl")) == 1000
    assert result["resumable"] and result["env_steps"] == 1008


def test_resource_pause_resumes_full_history_without_duplicate_training_frames(monkeypatch, tmp_path):
    env, task, folder, root = fake_collection(monkeypatch, tmp_path)
    env.pause_at = 3
    first = collection.collect_one(env, task, folder, root, [], 6)
    assert first["status"] == "paused_resource" and first["branch_steps"] == 3
    (root / "PAUSE").unlink()
    env.pause_at = None
    second = collection.collect_one(env, task, folder, root, [], 6)
    assert second["status"] == "censored_max_steps" and second["branch_steps"] == 6
    assert second["resume_method"] == "verified_full_action_replay"
    assert second["restore_overhead_steps"] == 11
    assert len(collection.read_jsonl(folder / "committed.jsonl")) == 6
