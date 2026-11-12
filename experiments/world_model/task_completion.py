"""Task-aware visual completion with same-episode, independently grounded identity."""

import json
import math
import re
from dataclasses import asdict

import numpy as np

from experiments.world_model.goal_verification import OBJECTS
from experiments.world_model.skill_contracts import GoalContract
from experiments.world_model.task_progress import StageEvidence
from experiments.world_model.visual_harness import parse_object


class QwenTaskAwareProgressVerifier:
    def __init__(self, vision, images):
        self.vision, self.images = vision, images
        self.scope, self.identities, self.execution_context = None, {}, {}
        self.last_trace = {}
        self.depth, self.depth_reference = None, None

    def set_execution_context(self, context):
        self.execution_context = context

    def set_depth(self, depth, reference, scale=1.0):
        self.depth = None if depth is None else np.asarray(depth, dtype=float).squeeze()*scale
        self.depth_reference = reference

    @staticmethod
    def _canonical_box(box):
        if (not isinstance(box, list) or len(box) != 4 or
                not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)):
            return None, "missing_or_invalid"
        values = [float(v) for v in box]
        if any(v < 0 or v > 1000 for v in values):
            return None, "out_of_range"
        if any(0 < v < 1 for v in values) and any(v > 1 for v in values):
            return None, "ambiguous_mixed_scale"
        unit = max(values) <= 1
        values = [v*1000 for v in values] if unit else values
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            return None, "invalid_box_order"
        return values, "unit_0_1" if unit else "normalized_0_1000"

    def _candidate_region_depth(self, reference, box):
        # A language-model box is not an object segmentation mask. Refuse a
        # single depth estimate when its central region contains distinct
        # foreground/background depths; even a coherent region is only an
        # estimate, never the reference-goal or certified object distance.
        if self.depth is None or reference != self.depth_reference or not isinstance(box, list) or len(box) != 4:
            return None, {"status": "unavailable_or_unbound"}
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
            return None, {"status": "invalid_box"}
        if self.depth.ndim != 2:
            return None, {"status": "invalid_depth_shape"}
        h, w = self.depth.shape
        x1, y1, x2, y2 = box
        crop = self.depth[int(h*(.8*y1+.2*y2)/1000):int(h*(.2*y1+.8*y2)/1000),
                          int(w*(.8*x1+.2*x2)/1000):int(w*(.2*x1+.8*x2)/1000)]
        values = crop[np.isfinite(crop) & (crop > .05) & (crop < 9.99)]
        if not crop.size or len(values) < crop.size*.5:
            return None, {"status": "insufficient_valid_depth"}
        p10, median, p90 = [float(x) for x in np.percentile(values, [10, 50, 90])]
        spread = p90-p10
        limit = max(.25, .25*median)  # conservative, uncalibrated region-consistency heuristic
        audit = {"status": "coherent_candidate_region_not_segmentation", "valid_fraction": len(values)/crop.size,
                 "p10_m": p10, "median_m": median, "p90_m": p90, "spread_m": spread,
                 "maximum_spread_m": limit, "not_a_certified_object_or_reference_goal_distance": True}
        if spread > limit:
            audit["status"] = "mixed_depths_no_single_target_range"
            return None, audit
        return median, audit

    def _ground(self, current, box, expected):
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
            return False, {"reason": "missing_or_invalid_target_box"}
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            return False, {"reason": "invalid_target_box"}
        if (x2-x1)*(y2-y1) >= 900000:
            return False, {"reason": "target_box_too_broad_to_ground_a_specific_object"}
        image = self.images.get(current)
        w, h = image.size
        crop = image.crop((int(w*x1/1000), int(h*y1/1000), int(w*x2/1000), int(h*y2/1000)))
        if min(crop.size) < 8 or float(np.mean(np.all(np.asarray(crop) == 0, axis=-1))) >= 0.9:
            return False, {"reason": "unusable_or_missing_mesh_crop"}
        prompt = ("Identify the visible object from this actual crop without assuming a task. "
            "Missing-mesh black triangles are unknown; require recognizable object structure. "
            "Return JSON with object_type (one of " + ", ".join([*OBJECTS, "unknown"]) +
            "), visible_structure (string), confidence (number 0..1). There are no default verdicts.")
        raw = self.vision(prompt, [("ACTUAL UNLABELLED CROP", crop)])
        value = parse_object(raw)
        confidence = float(value["confidence"])
        structure = value["visible_structure"]
        accepted = (value["object_type"] == expected and isinstance(structure, str) and bool(structure.strip())
                    and math.isfinite(confidence) and 0.8 <= confidence <= 1)
        audit = {**value, "raw_output": raw, "expected_category_not_in_prompt": expected}
        if accepted or value["object_type"] != "unknown" or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return accepted, audit
        # A tight crop may contain only part of a real object. The second view
        # supplies nearby context, while the task/category remains absent and
        # the model must identify the object in the first crop specifically.
        left, top, right, bottom = (int(w*x1/1000), int(h*y1/1000), int(w*x2/1000), int(h*y2/1000))
        dx, dy = right-left, bottom-top
        bounds = (max(0, left-dx), max(0, top-dy), min(w, right+dx), min(h, bottom+dy))
        if (bounds[2]-bounds[0])*(bounds[3]-bounds[1]) > .6*w*h:
            return False, {**audit, "context_skipped": "context_would_cover_most_of_the_frame"}
        context_prompt = ("Identify the object physically shown in the FIRST actual image crop. The second image is its "
            "surrounding actual context; use it only to understand structure connected to the first crop. "
            "Do not label a different nearby object. Missing-mesh black patches are unknown. Return JSON with "
            "object_type (one of " + ", ".join([*OBJECTS, "unknown"]) + "), visible_structure and confidence 0..1.")
        context_raw = self.vision(context_prompt, [("ACTUAL UNLABELLED CANDIDATE CROP", crop),
                                                    ("ACTUAL SURROUNDING CONTEXT", image.crop(bounds))])
        try:
            context_value = parse_object(context_raw)
            context_confidence = float(context_value["confidence"])
            context_structure = context_value["visible_structure"]
            accepted = (context_value["object_type"] == expected and isinstance(context_structure, str) and
                        bool(context_structure.strip()) and math.isfinite(context_confidence) and
                        0.8 <= context_confidence <= 1)
            audit["context_assisted"] = {**context_value, "raw_output": context_raw,
                                         "context_pixel_box": list(bounds), "accepted": accepted}
        except (ValueError, TypeError, KeyError) as exc:
            audit["context_assisted"] = {"raw_output": context_raw, "accepted": False, "error": str(exc)}
        return accepted, audit

    def _active_stage(self, progress, views):
        # A later goal must not turn an already completed route clause into a
        # negative. Keep its task text and confidence out of this model call.
        context = {"active_stage": asdict(progress.active),
            "completed_stages": [r["stage"]["instruction"] for r in progress.completed],
            "actual_execution": self.execution_context}
        prompt = ("Check ONLY this active navigation subtask against the ACTUAL first-person before/after views. "
            "The robot is the camera. Judge the requested motion or crossing, not an unmentioned later destination. "
            "A subtask can be complete even while the overall journey continues. Issued controls alone do not prove motion; "
            "use visual changes and measured feedback. Do not invent a room center, a new landmark or an extra stop. "
            "Return JSON with completed (boolean), confidence (0..1), reason (specific evidence or uncertainty). CONTEXT: "
            + json.dumps(context))
        raw = self.vision(prompt, views)
        value = parse_object(raw)
        confidence = float(value["confidence"])
        if not isinstance(value["completed"], bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid independent stage evidence")
        return {**value, "confidence": confidence, "raw_output": raw, "context": context}

    def __call__(self, progress, current_reference):
        scope = (progress.episode_id, progress.instruction)
        if scope != self.scope:
            self.scope, self.identities = scope, {}
        if self.execution_context.get("continuity_broken"):
            self.identities.clear()
        final = progress.stages[-1]
        contract = GoalContract.from_instruction(final.instruction)
        targets = [name for name, pattern in OBJECTS.items() if re.search(pattern, final.instruction, re.I)]
        expected = targets[0] if len(targets) == 1 else None
        step = progress.observations[current_reference]
        identities = {r: v for r, v in self.identities.items() if r in progress.observations and progress.observations[r] <= step}
        payload = {"original_instruction": progress.instruction, "active_stage": asdict(progress.active),
            "final_goal_contract": asdict(contract), "allowed_identity_references": list(identities),
            "completed_stages_with_actual_visual_support": progress.completed,
            "recent_actual_execution": self.execution_context,
            "recent_issued_controls_not_measured_motion": list(progress.executions.values())[-12:]}
        prompt = '''Judge ordered task progress from ACTUAL first-person views and execution feedback.
Judge the ACTIVE stage only for active_completed. Earlier route requirements still matter. Do not invent extra route stages.
Also identify the FINAL target independently of stage completion, so identity can be remembered within this episode.
For ENTER_ROOM, a previously grounded target plus visual continuity and actual crossing can establish room identity
even when that target is now behind the camera. A target out of view does NOT imply non-arrival.
For LOOK_AT require the target in the current view and the requested facing relation. STOP_AT_ENTRANCE may end at the threshold.
Seeing a landmark through a door is not itself proof of crossing. A bathroom visible through another door does not
prove the camera itself is in that bathroom. Issued movement is not measured movement. Never use a predicted future as evidence.
Missing-mesh triangles are not televisions or other objects. Admit uncertainty about identity or the camera's relation.
Front clearance is sensed free space, not distance to the requested goal. Local motion estimates have no target coordinate.
Use not_reached only with positive contradiction (wrong_room, before_entrance, blocked, wrong_target), not mere invisibility.
Return JSON with required fields: reason (concrete evidence or uncertainty), active_completed (boolean), target_visible
(boolean), target_bbox (integer-normalized coordinates 0..1000: [x1,y1,x2,y2], or null), identity_reference (actual current ref or one supplied identity ref, or null),
identity_continuity (boolean), relation_satisfied (boolean), camera_relation (inside_target_room, at_entrance,
at_requested_position, facing_target, outside_target_room, unknown), goal_verdict (reached, not_reached, unknown),
contradiction (wrong_room, before_entrance, blocked, wrong_target, none, unknown), confidence (0..1).
If the contract has additional_relations, also return established_relations (list) with only visually established relations.
No default verdicts. Target visibility and achieved position are independent. Do not claim completed from familiarity or policy STOP.
CONTEXT: ''' + json.dumps(payload)
        refs = [min(progress.observations, key=progress.observations.get), progress.anchor, *list(identities)[-2:]]
        for offset in [6, 1]:
            candidates = [(t, ref) for ref, t in progress.observations.items() if t <= step-offset]
            if candidates:
                refs.append(max(candidates)[1])
        refs = list(dict.fromkeys([*refs, current_reference]))[-7:]
        views = [(f"ACTUAL {'CURRENT' if r == current_reference else 'PAST'} VIEW reference={r}", self.images.get(r)) for r in refs]
        active = self._active_stage(progress, views) if progress.active.kind != "goal" else None
        raw = self.vision(prompt, views, max_new_tokens=700)
        self.last_trace = {"context": payload, "raw_output": raw, "identity_bank": self.identities.copy(),
                           "independent_stage_check": active}
        value = parse_object(raw)
        for name in ["active_completed", "target_visible", "identity_continuity", "relation_satisfied"]:
            if not isinstance(value[name], bool):
                raise ValueError("task evidence flags must be booleans")
        if value["goal_verdict"] not in {"reached", "not_reached", "unknown"}:
            raise ValueError("invalid goal verdict")
        confidence = float(value["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid task evidence confidence")
        raw_box = value["target_bbox"]
        box, box_format = self._canonical_box(raw_box)
        value["target_bbox"] = box
        grounded, audit = False, None
        if value["target_visible"]:
            if expected is not None:
                grounded, audit = self._ground(current_reference, value["target_bbox"], expected)
            else:
                grounded = confidence >= 0.8 and bool(value["reason"])
            if grounded:
                box = value["target_bbox"]
                if (not isinstance(box, list) or len(box) != 4 or
                        not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box) or
                        not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)):
                    box = None
                if box is not None:
                    width, height = self.images.get(current_reference).size
                    if ((box[2]-box[0])*width < 8000 or (box[3]-box[1])*height < 8000 or
                            (box[2]-box[0])*(box[3]-box[1]) >= 900000):
                        box = None
                self.identities[current_reference] = {"step": step, "category": expected, "box": box}
                self.identities = dict(list(self.identities.items())[-3:])
        identity_ref = current_reference if grounded else value["identity_reference"]
        # Past identity can support a target that has left the camera view.
        # It cannot excuse a conflicting current visibility claim whose actual
        # crop is a different object or cannot be independently recognized.
        current_visibility_failed = bool(value["target_visible"] and expected is not None and not grounded)
        identified = grounded or (not current_visibility_failed and value["identity_continuity"] and identity_ref in identities)
        if contract.require_current_target:
            identified = grounded
        relation = value["relation_satisfied"] and value["camera_relation"] == contract.camera_relation
        relation = relation and set(contract.additional_relations).issubset(value.get("established_relations", []))
        target_depth, depth_audit = (self._candidate_region_depth(current_reference, value["target_bbox"]) if grounded
                                     else (None, {"status": "target_not_grounded"}))
        proximity_depth_required = (contract.kind == "approach_target" and
            bool(re.search(r"\b(?:next\s+to|beside|near|close\s+to)\b", final.instruction, re.I)))
        if proximity_depth_required and target_depth is None:
            relation = False
        spatial_audit, spatial_negative = None, False
        if relation and grounded and contract.kind == "approach_target" and target_depth is not None:
            audit_context = {"requested_relation": final.instruction, "candidate_region_depth_estimate_m": target_depth,
                             "depth_region_not_an_object_segmentation_or_reference_goal_distance": True}
            spatial_raw = self.vision("Check whether the requested stopping relation is supported by this actual image and the "
                "candidate box's sensed depth. The box may include background; it is not object segmentation or a proven object distance. "
                "Seeing an object is not being beside it. Do not impose an arbitrary universal distance threshold. "
                "Return JSON with relation_supported (boolean), reason (concrete evidence), confidence (0..1). CONTEXT: "+json.dumps(audit_context),
                [("ACTUAL CURRENT VIEW", self.images.get(current_reference))])
            spatial_audit = parse_object(spatial_raw)
            if not isinstance(spatial_audit["relation_supported"], bool):
                raise ValueError("spatial audit requires an explicit boolean")
            audit_confidence = float(spatial_audit["confidence"])
            if not math.isfinite(audit_confidence) or not 0 <= audit_confidence <= 1:
                raise ValueError("invalid spatial confidence")
            relation = spatial_audit["relation_supported"] and audit_confidence >= .8
            spatial_negative = not spatial_audit["relation_supported"] and audit_confidence >= .8 and bool(spatial_audit["reason"])
            spatial_audit["raw_output"] = spatial_raw
        reached = value["goal_verdict"] == "reached" and identified and relation and confidence >= 0.8
        # An unseen target, a differently named object, or an obstruction is
        # not by itself proof that the target room has not been entered. A
        # before-entrance/wrong-room claim needs an independently grounded
        # target identity and an explicitly outside relation. The separate
        # depth audit can also contradict an approach-to-object relation.
        outside_grounded_target_room = (
            contract.kind in {"enter_room", "enter_and_look"}
            and value["contradiction"] in {"before_entrance", "wrong_room"}
            and identified
            and value["camera_relation"] in {"at_entrance", "outside_target_room"}
            and not value["relation_satisfied"]
            and isinstance(value["reason"], str)
            and bool(value["reason"].strip())
        )
        doorway_audit, outside_grounded_doorway = None, False
        if (progress.active.kind == "goal" and contract.kind == "stop_at_entrance" and
                value["goal_verdict"] == "not_reached" and value["contradiction"] == "before_entrance" and
                value["camera_relation"] == "outside_target_room" and not value["relation_satisfied"] and
                value["target_visible"] and isinstance(value["reason"], str) and value["reason"].strip()):
            outside_grounded_doorway, doorway_audit = self._ground(current_reference, value["target_bbox"], "door")
        negative = (value["goal_verdict"] == "not_reached" and confidence >= 0.8 and
                    (outside_grounded_target_room or outside_grounded_doorway))
        verdict = "reached" if reached else "not_reached" if spatial_negative or negative else "unknown"
        completed = reached if active is None else active["completed"]
        evidence_confidence = confidence if active is None else active["confidence"]
        evidence_reason = str(value["reason"] if active is None else active["reason"])
        evidence = StageEvidence(progress.active.stage_id, progress.anchor, current_reference, bool(completed),
            bool(grounded), evidence_confidence, evidence_reason, bool(identified), identity_ref if identified else None, verdict)
        self.last_trace = {"context": payload, "raw_output": raw, "grounding_audit": audit,
            "identity_bank": self.identities.copy(), "goal_verdict": verdict, "evidence": asdict(evidence),
            "candidate_region_depth_estimate_m": target_depth, "candidate_depth_audit": depth_audit, "spatial_audit": spatial_audit,
            "proximity_depth_required": proximity_depth_required,
            "current_visibility_claim_failed_grounding": current_visibility_failed,
            "raw_target_bbox": raw_box, "normalized_target_bbox": value["target_bbox"], "target_box_format": box_format,
            "current_target_visibility_required": contract.require_current_target,
            "grounded_outside_relation_supports_rejection": outside_grounded_target_room,
            "grounded_doorway_supports_rejection": bool(outside_grounded_doorway),
            "doorway_rejection_audit": doorway_audit,
            "independent_stage_check": active}
        return evidence
