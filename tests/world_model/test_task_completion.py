import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.world_model.stateful_controller import StatefulEpisodeController
from experiments.world_model.task_completion import QwenTaskAwareProgressVerifier
from experiments.world_model.task_progress import QwenTaskDecomposer, TaskProgress, TaskStage
from experiments.world_model.visual_harness import ObservationImages
from scripts.calibrate_harness_completion import set_case_depth


def value(*, visible=False, reached=False, ref=None, relation="inside_target_room", contradiction="none", verdict=None):
    return {"reason": "actual target structure and camera transition", "active_completed": reached,
        "target_visible": visible, "target_bbox": [100, 100, 900, 900] if visible else None,
        "identity_reference": ref, "identity_continuity": ref is not None,
        "relation_satisfied": reached, "camera_relation": relation,
        "goal_verdict": verdict or ("reached" if reached else "unknown"), "contradiction": contradiction, "confidence": .95}


def fixture(instruction="enter the room with the television", episode="ep", size=(40,40)):
    images = ObservationImages()
    for ref in ["start", "seen", "inside", "turned", "new"]:
        images.put(ref, np.full((*size, 3), 120, dtype=np.uint8))
    tracker = TaskProgress(episode_id=episode, instruction=instruction,
        stages=[TaskStage("goal", instruction, "requested relation", "goal")], initial_observation="start",
        require_current_target=False)
    return images, tracker


def record(tracker, reference, step):
    for t in range(max(tracker.observations.values()), step):
        tracker.executed(t, environment_step_returned=True, rgb_changed=True,
                         action={"action": "MOVE_FORWARD"})
    tracker.observe(reference, step)


def test_grounded_target_can_leave_view_after_actual_entry_without_blocking_stop():
    images, p = fixture()
    replies = iter([value(visible=True), {"object_type": "television", "visible_structure": "bezel and stand", "confidence": .95},
                    value(reached=True, ref="seen"), value(reached=True, ref="seen")])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    assert not p.consider(verifier(p, "seen"))
    record(p, "inside", 6)
    evidence = verifier(p, "inside")
    assert evidence.completed and not evidence.target_visible and evidence.target_identified
    assert not p.consider(evidence)
    record(p, "turned", 9)
    assert p.consider(verifier(p, "turned")) and p.accept_global_stop()


def test_previous_identity_cannot_mask_a_failed_current_visibility_claim():
    images, p = fixture()
    wrong_current = value(visible=True, reached=True, ref='seen')
    replies = iter([value(visible=True), {'object_type':'television','visible_structure':'screen and bezel','confidence':.95},
                    wrong_current, {'object_type':'cabinet','visible_structure':'wooden panel and handles','confidence':.95},
                    value(reached=True, ref='seen')])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, 'seen', 3)
    verifier(p, 'seen')
    record(p, 'inside', 6)
    conflict = verifier(p, 'inside')
    assert not conflict.completed and not conflict.target_identified and conflict.verdict == 'unknown'
    assert verifier.last_trace['current_visibility_claim_failed_grounding']
    record(p, 'turned', 9)
    after_target_leaves_view = verifier(p, 'turned')
    assert after_target_leaves_view.completed and after_target_leaves_view.target_identified


def test_goal_identity_does_not_leak_into_a_new_episode():
    images, p = fixture()
    replies = iter([value(visible=True), {"object_type": "television", "visible_structure": "bezel", "confidence": .9},
                    value(reached=True, ref="seen")])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    verifier(p, "seen")
    _images, other = fixture(episode="other")
    record(other, "new", 3)
    evidence = verifier(other, "new")
    assert not evidence.completed and not evidence.target_identified
    assert evidence.verdict == "unknown"


def test_look_at_target_cannot_complete_using_only_a_past_target_view():
    images, p = fixture("look at the television")
    replies = iter([value(visible=True, relation="unknown"),
                    {"object_type": "television", "visible_structure": "screen and bezel", "confidence": .9},
                    value(reached=True, ref="seen", relation="facing_target")])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    verifier(p, "seen")
    record(p, "turned", 6)
    evidence = verifier(p, "turned")
    assert not evidence.completed and evidence.verdict == "unknown"


def test_invisibility_alone_cannot_be_a_confident_nonarrival():
    images, p = fixture()
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(value(verdict="not_reached")), images)
    record(p, "new", 3)
    assert verifier(p, "new").verdict == "unknown"


@pytest.mark.parametrize("contradiction", ["wrong_target", "wrong_room", "before_entrance", "blocked"])
def test_named_contradiction_without_grounded_target_does_not_prove_nonarrival(contradiction):
    images, p = fixture()
    reply = value(verdict="not_reached", contradiction=contradiction, relation="outside_target_room")
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(reply), images)
    record(p, "new", 3)
    assert verifier(p, "new").verdict == "unknown"


def test_grounded_television_visible_from_before_its_entrance_can_refute_entry():
    images, p = fixture()
    replies = iter([value(visible=True, verdict="not_reached", contradiction="before_entrance",
                          relation="outside_target_room"),
                    {"object_type": "television", "visible_structure": "screen and bezel", "confidence": .95}])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    evidence = verifier(p, "seen")
    assert evidence.target_identified and not evidence.completed
    assert evidence.verdict == "not_reached"


@pytest.mark.parametrize("contradiction", ["wrong_target", "blocked"])
def test_grounded_target_does_not_turn_an_unproven_obstruction_into_nonarrival(contradiction):
    images, p = fixture()
    replies = iter([value(visible=True, verdict="not_reached", contradiction=contradiction,
                          relation="outside_target_room"),
                    {"object_type": "television", "visible_structure": "screen and bezel", "confidence": .95}])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    assert verifier(p, "seen").verdict == "unknown"


def test_an_outside_label_conflicting_with_the_claimed_relation_does_not_refute_entry():
    images, p = fixture()
    reply = value(visible=True, reached=True, verdict="not_reached", contradiction="before_entrance",
                  relation="outside_target_room")
    replies = iter([reply, {"object_type": "television", "visible_structure": "screen and bezel", "confidence": .95}])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    assert verifier(p, "seen").verdict == "unknown"


def test_independently_grounded_door_can_show_the_doorway_is_still_ahead():
    images, p = fixture("stop in the doorway")
    replies = iter([value(visible=True, verdict="not_reached", contradiction="before_entrance",
                          relation="outside_target_room"),
                    {"object_type": "door", "visible_structure": "door leaf and frame", "confidence": .95}])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    assert verifier(p, "seen").verdict == "not_reached"
    assert verifier.last_trace["grounded_doorway_supports_rejection"]


def test_unrecognized_door_does_not_prove_the_requested_doorway_is_ahead():
    images, p = fixture("stop in the doorway")
    replies = iter([value(visible=True, verdict="not_reached", contradiction="before_entrance",
                          relation="outside_target_room"),
                    {"object_type": "window", "visible_structure": "glass and frame", "confidence": .95}])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    assert verifier(p, "seen").verdict == "unknown"


def test_unusable_box_cannot_direct_a_recovery_view_toward_an_imagined_bearing():
    images, p = fixture("stop in the doorway")
    reply = value(visible=True, reached=True, relation="at_entrance")
    reply["target_bbox"] = [.88, 0., 1., .25]  # Even after normalization, this is under 8 pixels wide.
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(reply), images)
    record(p, "seen", 3)
    assert verifier(p, "seen").completed
    controller = SimpleNamespace(progress_verifier=verifier, feedback=SimpleNamespace(hfov=120.),
                                 current_reference="seen")
    assert StatefulEpisodeController._target_bearing(controller) is None


@pytest.mark.parametrize(('box','box_format'), [
    ([.85,.35,1.,.65], 'unit_0_1'),
    ([850,350,1000,650], 'normalized_0_1000'),
])
def test_target_box_coordinate_formats_ground_the_same_actual_object(box,box_format):
    images, p = fixture('stop beside the table',size=(120,160))
    reply=value(visible=True,reached=True,relation='at_requested_position')
    reply['target_bbox']=box
    replies=iter([reply,{'object_type':'table','visible_structure':'tabletop and leg','confidence':.95},
                  {'relation_supported':True,'reason':'the table surface is beside the camera','confidence':.95}])
    verifier=QwenTaskAwareProgressVerifier(lambda *_a,**_k:json.dumps(next(replies)),images)
    record(p,'seen',3)
    verifier.set_depth(np.full((120,160),.6),'seen')
    evidence=verifier(p,'seen')
    assert evidence.completed and evidence.target_identified
    assert verifier.last_trace['target_box_format']==box_format
    assert verifier.last_trace['normalized_target_bbox']==[850.,350.,1000.,650.]
    assert verifier.last_trace['candidate_region_depth_estimate_m']==pytest.approx(.6)
    assert verifier.last_trace['candidate_depth_audit']['not_a_certified_object_or_reference_goal_distance']


def test_full_frame_box_cannot_ground_a_specific_target():
    images,p=fixture('stop beside the table',size=(120,160))
    reply=value(visible=True,reached=True,relation='at_requested_position')
    reply['target_bbox']=[0.,0.,1.,1.]
    verifier=QwenTaskAwareProgressVerifier(lambda *_a,**_k:json.dumps(reply),images)
    record(p,'seen',3)
    evidence=verifier(p,'seen')
    assert not evidence.completed and evidence.verdict=='unknown'
    assert verifier.last_trace['grounding_audit']['reason']=='target_box_too_broad_to_ground_a_specific_object'


def test_mixed_coordinate_scales_remain_unknown():
    images,p=fixture('stop beside the table',size=(120,160))
    reply=value(visible=True,reached=True,relation='at_requested_position')
    reply['target_bbox']=[.85,350.,1.,650.]
    verifier=QwenTaskAwareProgressVerifier(lambda *_a,**_k:json.dumps(reply),images)
    record(p,'seen',3)
    evidence=verifier(p,'seen')
    assert not evidence.completed and evidence.verdict=='unknown'
    assert verifier.last_trace['target_box_format']=='ambiguous_mixed_scale'


def test_context_can_ground_a_partial_actual_table_without_task_label_in_the_prompt():
    images,p=fixture('stop beside the table',size=(120,160))
    reply=value(visible=True,reached=True,relation='at_requested_position')
    reply['target_bbox']=[.85,.35,1.,.65]
    replies=iter([reply,{'object_type':'unknown','visible_structure':'a dark flat edge','confidence':.6},
        {'object_type':'table','visible_structure':'a tabletop connected to a supporting leg','confidence':.85},
        {'relation_supported':True,'reason':'the visible table surface is beside the camera','confidence':.9}])
    calls=[]
    def vision(prompt,views,**_kwargs):
        calls.append((prompt,views))
        return json.dumps(next(replies))
    verifier=QwenTaskAwareProgressVerifier(vision,images)
    record(p,'seen',3)
    verifier.set_depth(np.full((120,160),.6),'seen')
    evidence=verifier(p,'seen')
    assert evidence.completed and evidence.target_identified
    assert verifier.last_trace['grounding_audit']['context_assisted']['accepted']
    assert len(calls[2][1])==2 and 'stop beside the table' not in calls[2][0]
    assert verifier.last_trace['candidate_region_depth_estimate_m']==pytest.approx(.6)


def test_context_does_not_promote_a_different_object_into_a_television():
    images,p=fixture(size=(120,160))
    reply=value(visible=True,reached=True)
    reply['target_bbox']=[.1,.4,.15,.5]
    replies=iter([reply,{'object_type':'unknown','visible_structure':'dark edge','confidence':.5},
                  {'object_type':'cabinet','visible_structure':'cabinet door and handle','confidence':.95}])
    verifier=QwenTaskAwareProgressVerifier(lambda *_a,**_k:json.dumps(next(replies)),images)
    record(p,'seen',3)
    evidence=verifier(p,'seen')
    assert not evidence.completed and evidence.verdict=='unknown' and not evidence.target_identified
    assert not verifier.last_trace['grounding_audit']['context_assisted']['accepted']


def test_explicit_proximity_remains_unknown_without_target_surface_depth():
    images,p=fixture('stop beside the table',size=(120,160))
    reply=value(visible=True,reached=True,relation='at_requested_position')
    reply['target_bbox']=[.85,.35,1.,.65]
    replies=iter([reply,{'object_type':'table','visible_structure':'tabletop and leg','confidence':.95}])
    verifier=QwenTaskAwareProgressVerifier(lambda *_a,**_k:json.dumps(next(replies)),images)
    record(p,'seen',3)
    evidence=verifier(p,'seen')
    assert evidence.target_identified and not evidence.completed and evidence.verdict=='unknown'
    assert verifier.last_trace['proximity_depth_required']


def test_mixed_foreground_background_depth_cannot_certify_proximity():
    images,p=fixture('stop beside the table',size=(120,160))
    reply=value(visible=True,reached=True,relation='at_requested_position')
    reply['target_bbox']=[.5,.25,1.,.75]
    replies=iter([reply,{'object_type':'table','visible_structure':'tabletop and leg','confidence':.95}])
    verifier=QwenTaskAwareProgressVerifier(lambda *_a,**_k:json.dumps(next(replies)),images)
    record(p,'seen',3)
    depth=np.full((120,160),2.)
    depth[:,80:120]=.6
    verifier.set_depth(depth,'seen')
    evidence=verifier(p,'seen')
    assert evidence.target_identified and not evidence.completed and evidence.verdict=='unknown'
    assert verifier.last_trace['candidate_region_depth_estimate_m'] is None
    assert verifier.last_trace['candidate_depth_audit']['status']=='mixed_depths_no_single_target_range'
    assert verifier.last_trace['spatial_audit'] is None


def test_calibration_depth_must_be_the_hashed_current_observation(tmp_path):
    images,p=fixture('stop beside the table',size=(120,160))
    path=tmp_path/'actual_depth.npy'
    np.save(path,np.full((120,160),.06))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    binding={'reference':'seen','path':str(path),'sha256':digest,'metres_per_depth_unit':10.}
    case={'current_reference':'seen','current_depth':binding}
    verifier=QwenTaskAwareProgressVerifier(lambda *_a,**_k:'{}',images)
    assert set_case_depth(verifier,case)
    assert np.nanmedian(verifier.depth)==pytest.approx(.6)
    with pytest.raises(ValueError,match='current actual frame'):
        set_case_depth(verifier,{**case,'current_reference':'later'})
    np.save(path,np.full((10,10),.06))
    wrong_shape={**binding,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError,match='image dimensions'):
        set_case_depth(verifier,{**case,'current_depth':wrong_shape})
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='binding changed'):
        set_case_depth(verifier,case)


def test_unrecognized_black_artifact_cannot_become_a_persistent_target_identity():
    images, p = fixture()
    replies = iter([value(visible=True, reached=True),
                    {"object_type": "unknown", "visible_structure": "missing triangle", "confidence": .99}])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    assert not verifier(p, "seen").completed and not verifier.identities


def test_compound_enter_and_look_goal_keeps_both_relations():
    images, p = fixture("enter the room and look at the television")
    replies = iter([value(visible=True, reached=True, relation="facing_target"),
                    {"object_type": "television", "visible_structure": "screen and bezel", "confidence": .95}])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    assert not verifier(p, "seen").completed


def test_literal_decomposition_cannot_invent_a_room_center_or_skip_route_clauses():
    def vision(*_a, **_k):
        return json.dumps({"stages": [{"instruction": "go to the center of the room", "kind": "navigate"},
                                      {"instruction": "enter the television room", "kind": "goal"}]})
    decomposer = QwenTaskDecomposer(vision, require_verbatim=True)
    with pytest.raises(ValueError, match="invented"):
        decomposer("Cross the room and enter the television room.")
    decomposer.vision = lambda *_a, **_k: json.dumps({"stages": [{"instruction": "enter the television room", "kind": "goal"}]})
    with pytest.raises(ValueError, match="omitted"):
        decomposer("Cross the room and enter the television room.")


def test_literal_decomposition_preserves_ordered_clauses_and_turn_type():
    data = {"stages": [{"instruction": "Turn left", "kind": "navigate"},
                       {"instruction": "enter the room", "kind": "goal"}]}
    stages = QwenTaskDecomposer(lambda *_a, **_k: json.dumps(data), require_verbatim=True)("Turn left and enter the room.")
    assert [s.kind for s in stages] == ["turn", "goal"]
    assert stages[-1].instruction == "enter the room"


def test_explicit_sequence_cannot_be_claimed_as_one_verified_stage():
    instruction = "Turn left then enter the room."
    data = {"stages": [{"instruction": instruction, "kind": "goal"}]}
    decomposer = QwenTaskDecomposer(lambda *_a, **_k: json.dumps(data), require_verbatim=True)
    with pytest.raises(ValueError, match="undivided stage"):
        decomposer(instruction)


def test_current_target_surface_depth_can_reject_a_merely_visible_distant_object():
    images, p = fixture("stop beside the television")
    replies = iter([value(visible=True, reached=True, relation="at_requested_position"),
                    {"object_type": "television", "visible_structure": "screen and bezel", "confidence": .95},
                    {"relation_supported": False, "reason": "the actual surface is still four metres away", "confidence": .95}])
    prompts = []
    def vision(prompt, *_a, **_k):
        prompts.append(prompt)
        return json.dumps(next(replies))
    verifier = QwenTaskAwareProgressVerifier(vision, images)
    record(p, "seen", 3)
    verifier.set_depth(np.full((40, 40), 4.), "seen")
    evidence = verifier(p, "seen")
    assert not evidence.completed and evidence.verdict == "not_reached"
    assert '"candidate_region_depth_estimate_m": 4.0' in prompts[-1]
    assert 'not object segmentation' in prompts[-1]


def test_route_stage_completion_is_independent_of_later_target_visibility_and_confidence():
    images, p = fixture()
    p.stages = (TaskStage("cross", "Walk across the room", "Actual crossing", "navigate"), *p.stages)
    final = value()
    final["confidence"] = .1
    replies = iter([{"completed": True, "confidence": .95, "reason": "Actual views establish the crossing"}, final])
    prompts = []
    def vision(prompt, *_a, **_k):
        prompts.append(prompt)
        return json.dumps(next(replies))
    verifier = QwenTaskAwareProgressVerifier(vision, images)
    record(p, "seen", 3)
    evidence = verifier(p, "seen")
    assert evidence.completed and evidence.confidence == .95
    assert not evidence.target_identified and evidence.verdict == "unknown"
    assert "television" not in prompts[0]
    assert verifier.last_trace["independent_stage_check"]["completed"]


def test_positive_final_goal_does_not_complete_an_unfinished_required_route_clause():
    images, p = fixture("stop in the doorway")
    p.stages = (TaskStage("turn", "Turn left", "Actual turn", "turn"), *p.stages)
    replies = iter([{"completed": False, "confidence": .9, "reason": "No actual left turn yet"},
                    value(visible=True, reached=True, relation="at_entrance")])
    verifier = QwenTaskAwareProgressVerifier(lambda *_a, **_k: json.dumps(next(replies)), images)
    record(p, "seen", 3)
    evidence = verifier(p, "seen")
    assert not evidence.completed and not p.consider(evidence) and not p.accept_global_stop()
