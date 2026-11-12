import json

import numpy as np

from experiments.world_model.goal_verification import QwenStrictProgressVerifier
from experiments.world_model.task_progress import TaskProgress, TaskStage
from experiments.world_model.visual_harness import ObservationImages


def tracker():
    p = TaskProgress(episode_id="ep", instruction="enter the room with the television",
        stages=[TaskStage("goal", "enter the room with the television", "camera inside TV room", "goal")], initial_observation="start")
    p.observe("now", 6)
    images = ObservationImages()
    images.put("start", np.full((40, 40, 3), 100, dtype=np.uint8))
    images.put("now", np.full((40, 40, 3), 120, dtype=np.uint8))
    return p, images


def verdict(relation):
    return {"reason": "observed camera relation", "target_visible": True, "relation_satisfied": True,
            "camera_relation": relation, "target_bbox": [100, 100, 900, 900], "confidence": 0.95}


def test_visible_tv_from_threshold_does_not_establish_entering_its_room():
    p, images = tracker()
    v = QwenStrictProgressVerifier(lambda *_: json.dumps(verdict("at_entrance")), images)
    assert not v(p, "now").completed


def test_crop_without_task_prompt_can_reject_a_hallucinated_target():
    p, images = tracker()
    calls = []

    def vision(prompt, *_):
        calls.append(prompt)
        return json.dumps(verdict("inside_target_room") if len(calls) == 1 else
            {"object_type": "unknown", "visible_structure": "missing mesh", "confidence": 0.99})

    v = QwenStrictProgressVerifier(vision, images)
    assert not v(p, "now").completed
    assert "enter the room" not in calls[1] and "requested target" not in calls[1]
    assert '"target_visible":false' not in calls[0] and '"confidence":0.0' not in calls[0]


def test_relation_and_independently_recognized_object_both_required():
    p, images = tracker()
    responses = iter([verdict("inside_target_room"),
        {"object_type": "television", "visible_structure": "rectangular bezel, screen and stand", "confidence": 0.9}])
    v = QwenStrictProgressVerifier(lambda *_: json.dumps(next(responses)), images)
    assert v(p, "now").completed


def test_explicit_stop_in_doorway_accepts_threshold_instead_of_demanding_room_entry():
    p, images = tracker()
    p.stages = (TaskStage("goal", "Stop when you're in the doorway", "camera in doorway", "goal"),)
    v = QwenStrictProgressVerifier(lambda *_: json.dumps(verdict("at_entrance")), images)
    assert v(p, "now").completed
